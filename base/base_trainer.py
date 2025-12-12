import glob
import torch
from torch.utils import tensorboard
from tqdm import tqdm

from utils import helpers
from utils.losses import CE_loss, Alg_loss
import utils.lr_scheduler
import os, json, math, sys, datetime
from models import *
from types import SimpleNamespace 
from pathlib import Path
import datetime


def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])

class BaseTrainer:
    def __init__(self, model, resume, config, iters_per_epoch):
        self.model = model
        self.config = config
        self.iters_per_epoch = iters_per_epoch
        self.do_validation = self.config['trainer']['val']
        self.start_epoch = 1
        self.process = self.config['trainer']['process']
        # SETTING THE DEVICE
        self.device, availble_gpus = self._get_available_devices(self.config['n_gpu'])
        self.model = torch.nn.DataParallel(self.model)
        self.model.to(self.device)
        print(self.device)
        # CONFIGS
        cfg_trainer = self.config['trainer']
        
        # New addition
        self.sub_epochs = cfg_trainer['sub_epochs']
        self.epochs = sum(self.sub_epochs)

        self.epoch_stage1 = self.sub_epochs[0]
        self.epoch_stage2 = self.sub_epochs[1]
        self.epoch_stage3 = self.sub_epochs[2]

        self.save_period = cfg_trainer['save_period']

        # OPTIMIZER
        trainable_params = [{'params': filter(lambda p:p.requires_grad, self.model.module.get_other_params())},
                            {'params': filter(lambda p:p.requires_grad, self.model.module.get_backbone_params()), 
                            'lr': config['optimizer']['args']['lr'] / 10}]
        # SGD optimizer
        self.optimizer = get_instance(torch.optim, 'optimizer', config, trainable_params)
        model_params = sum([i.shape.numel() for i in list(model.parameters())])
        opt_params = sum([i.shape.numel() for j in self.optimizer.param_groups for i in j['params']])
        assert opt_params == model_params, 'some params are missing in the opt'

        self.lr_scheduler = getattr(utils.lr_scheduler, config['lr_scheduler'])(optimizer=self.optimizer, num_epochs=self.epoch_stage1,
                                        iters_per_epoch=iters_per_epoch)

        # MONITORING
        self.mnt_curr = 0
        self.mnt_best = 0
        self.improved = False
        self.min_nf_train_loss = 1e9
        self.improve_nf = True

        # MONITORING
        self.nf_mnt_curr = 0
        self.nf_mnt_best = 0
        self.nf_improved = False

        # From which stage to start training
        self.curr_stage = ""

        # CHECKPOINTS & TENSOBOARD
        #date_time = datetime.datetime.now().strftime('%m-%d_%H-%M')
        run_name = config['experim_name']
        self.note_name= config['note_name']
        # note_name is not needed in stages 1 and 2
        self.checkpoint_dir = os.path.join(cfg_trainer['save_dir'], run_name)
        helpers.dir_exists(self.checkpoint_dir)

         
        if resume: self._resume_checkpoint(resume)
        
        # nf decoder
        self.nf_decoders = None
        

    def _get_available_devices(self, n_gpu):
        sys_gpu = torch.cuda.device_count()
        if sys_gpu == 0:
            print ('No GPUs detected, using the CPU')
            n_gpu = 0
        elif n_gpu > sys_gpu:
            print (f'Nbr of GPU requested is {n_gpu} but only {sys_gpu} are available')
            n_gpu = sys_gpu
            
        device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
        available_gpus = list(range(n_gpu))

        return device, available_gpus


    def train(self):
        # first stage train
        if 1 in self.process:
            config_1_dir = os.path.join(self.checkpoint_dir, 'stage1')
            os.makedirs(config_1_dir, exist_ok=True)
            with open(os.path.join(config_1_dir, 'config.json'), 'w') as handle:
                json.dump(self.config, handle, indent=4, sort_keys=True)
            print("="*50, "Start stage ONE", "="*50)
            self.curr_stage = "one"
            for epoch in range(self.start_epoch, self.epoch_stage1+1):

                self._train_epoch(epoch)

                if self.do_validation and epoch % self.config['trainer']['val_per_epochs'] == 0:
                    self.mnt_curr = self._valid_epoch(epoch)
                # CHECKING IF THIS IS THE BEST MODEL (ONLY FOR VAL)
                if epoch % self.config['trainer']['val_per_epochs'] == 0:
                    self.improved = (self.mnt_curr > self.mnt_best)
                    self.mnt_best = self.mnt_curr if self.improved else self.mnt_best

                # SAVE CHECKPOINT
                if epoch % self.save_period == 0:
                    stage_one_best_checkpoint = self._save_checkpoint(epoch, save_best=self.improved, stage="stage1")
            print("="*50, "End stage ONE", "="*50)
        # second stage train
        if 2 in self.process:
            # Training
            print("="*50, "Start stage TWO", "="*50)
            self.start_epoch = 1
            # Load model and freeze parameters
            checkpoint = torch.load(self.config['trainer']['best_stage_one_model'], map_location='cuda')
            print(f"checkpoint loaded: {self.config['trainer']['best_stage_one_model']}")
            self.model.module.load_state_dict(checkpoint['state_dict'], strict=False)
            for param in self.model.parameters():
                param.requires_grad = False

            self.curr_stage = "two"
            for epoch in tqdm(range(self.start_epoch, self.epoch_stage2+1)):
                self._train_epoch(epoch, stage='two', nf_decoders=self.nf_decoders)
                # if not self.improve_nf:
                #     break
            # Testing
            # print("Start: nf_test")
            # self._nf_load_weights(self.config['trainer']['best_stage_two_model'])
            # for decoder in self.nf_decoders:
            #     for param in decoder.parameters():
            #         param.requires_grad = False
            # _ = self._test_second_stage_train(epoch, self.nf_decoders)
            #
            # # Unfreeze parameters
            # for param in self.model.parameters():
            #     param.requires_grad = True
            #
            # print("=" * 50, "End stage TWO", "=" * 50)
            # self._components_tongji(self.nf_decoders)
        # generate pseudo labels
        if 3 in self.process:
            self.checkpoint_dir=os.path.join(self.checkpoint_dir, self.note_name)
            os.makedirs(self.checkpoint_dir, exist_ok=True)

            checkpoint = torch.load(self.config['trainer']['best_stage_one_model'], map_location='cuda')
            print(f"checkpoint loaded: {self.config['trainer']['best_stage_one_model']}")
            self.model.module.load_state_dict(checkpoint['state_dict'], strict=False)
            for param in self.model.parameters():
                param.requires_grad = False

            self._nf_load_weights(self.config['trainer']['best_stage_two_model'])
            print(f"checkpoint loaded: {self.config['trainer']['best_stage_two_model']}")
            for decoder in self.nf_decoders:
                for param in decoder.parameters():
                    param.requires_grad = False

            self._generate_pseudo_labels()
        # third stage train
        if 4 in self.process:
            print("=" * 50, "Start stage THREE", "=" * 50)

            # Set original model parameters to be optimizable
            for param in self.model.parameters():
                param.requires_grad = True
            # Save config
            self.checkpoint_dir=os.path.join(self.checkpoint_dir, self.note_name)
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            # with open(os.path.join(self.checkpoint_dir, "config.json"), 'w', encoding='utf-8') as f:
            #     json.dump(self.config, f, ensure_ascii=False, indent=4)

            # Reset optimizer
            self._reset_optimizer()

            self.curr_stage = "three"
            self.method = "third"
            self.start_epoch = 1
            for epoch in range(self.start_epoch, self.epoch_stage3+1):
                self._train_epoch(epoch, stage='three')

                if self.do_validation and epoch % self.config['trainer']['val_per_epochs'] == 0:
                    self.mnt_curr = self._valid_epoch(epoch)
                # CHECKING IF THIS IS THE BEST MODEL (ONLY FOR VAL)
                if epoch % self.config['trainer']['val_per_epochs'] == 0:
                    self.improved = (self.mnt_curr > self.mnt_best)
                    self.mnt_best = self.mnt_curr if self.improved else self.mnt_best

                if epoch % self.save_period == 0:
                    self._save_checkpoint(epoch, save_best=self.improved, stage="stage3")
            print("=" * 50, "End stage THREE", "=" * 50)
            # remove peudo labels
            fake_label_dir = self.config["fake_labels_dir"]
            for f in glob.glob(os.path.join(fake_label_dir, "Label_batch_*.pt")):
                os.remove(f)
            for f in glob.glob(os.path.join(fake_label_dir, "noLabel_batch_*.pt")):
                os.remove(f)

        if 5 in self.process:
            checkpoint = torch.load(self.config['trainer']['best_stage_one_model'], map_location='cuda')
            print(f"checkpoint loaded: {self.config['trainer']['best_stage_one_model']}")
            self.model.module.load_state_dict(checkpoint['state_dict'])
            for param in self.model.parameters():
                param.requires_grad = False
            self._nf_load_weights(self.config['trainer']['best_stage_two_model'])
            for decoder in self.nf_decoders:
                for param in decoder.parameters():
                    param.requires_grad = False
            self._components_tongji(self.nf_decoders)


    def _reset_optimizer(self):
        trainable_params = [{'params': filter(lambda p: p.requires_grad, self.model.module.get_other_params())},
                            {'params': filter(lambda p: p.requires_grad, self.model.module.get_backbone_params()),
                             'lr': self.config['optimizer']['args']['lr'] / 10}]
        self.optimizer = get_instance(torch.optim, 'optimizer', self.config, trainable_params)
        model_params = sum([i.shape.numel() for i in list(self.model.parameters())])
        opt_params = sum([i.shape.numel() for j in self.optimizer.param_groups for i in j['params']])
        assert opt_params == model_params, 'Stage 3 some params are missing in the opt'

        self.lr_scheduler = getattr(utils.lr_scheduler, self.config['lr_scheduler'])(optimizer=self.optimizer,
                                                                                num_epochs=self.epoch_stage3,
                                                                                iters_per_epoch=self.iters_per_epoch)
    def _save_checkpoint(self, epoch, stage, save_best=False):
        state = {
            'arch': type(self.model).__name__,
            'epoch': epoch,
            'state_dict': self.model.module.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        weight_dir = Path(self.checkpoint_dir) / stage
        weight_dir.mkdir(exist_ok=True, parents=True)
        filename = weight_dir / 'checkpoint_thr-{}.pth'.format(self.config['model']['confidence_thr'])
        print (f'\nSaving a checkpoint: {filename} ...') 
        torch.save(state, filename)

        if save_best:
            filename = weight_dir / 'best_model_thr-{}.pth'.format(self.config['model']['confidence_thr'])
            torch.save(state, filename)
            print ("Saving current best: best_model.pth")
        return filename

    def _resume_checkpoint(self, resume_path):
        print (f'Loading checkpoint : {resume_path}')
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        try:
            self.model.module.load_state_dict(checkpoint['state_dict'])
        except Exception as e:
            print (f'Error when loading: {e}')
            self.model.module.load_state_dict(checkpoint['state_dict'], strict=False)
            
    def _nf_save_weights(self, epoch, stage = "stage2_nf", save_best=False):
        weight_dir = Path(self.checkpoint_dir) / stage / 'nf'
        weight_dir.mkdir(exist_ok=True, parents=True)
        state = {'decoder_state_dict': [decoder.state_dict() for decoder in self.nf_decoders]}
        filename = 'nf_epoch_last.pth'
        path = weight_dir / filename
        torch.save(state, path)
        print('Saving weights to {}'.format(filename))
        
        if save_best:
            filename = weight_dir / 'best_model_nf_decoders.pth'
            torch.save(state, filename)
            print ("Saving current NF BEST: best_model.pth")


    def _nf_load_weights(self, nf_resume_path):        
        state = torch.load(nf_resume_path)         
        [decoder.load_state_dict(s, strict=False) for decoder, s in zip(self.nf_decoders, state['decoder_state_dict'])]
        print('Loading weights from {}'.format(nf_resume_path))



    def _train_epoch(self, epoch, state='one'):
        raise NotImplementedError
    
    def _test_second_stage_train(self, epoch, nf_decoders):
        raise NotImplementedError

    def _valid_epoch(self, epoch):
        raise NotImplementedError

    def _eval_metrics(self, output, target):
        raise NotImplementedError
