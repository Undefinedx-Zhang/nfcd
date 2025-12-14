import os
import glob
from pathlib import Path

import torch
import json
import time, random, cv2, sys
from math import ceil
import numpy as np
from itertools import cycle
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils import data
from torchvision.utils import make_grid
from torchvision import transforms
from base import BaseTrainer
from utils.helpers import colorize_mask
from utils.metrics import eval_metrics, AverageMeter
from tqdm import tqdm
from PIL import Image
from utils.helpers import DeNormalize
from utils.lr_scheduler import adjust_learning_rate, warmup_learning_rate
from utils.losses import get_logp, t2np
from utils.metrics import inter_over_union, compute_intersection_union
# from utils.visualize import save_batch_prob_and_labels
from utils.idea import refine_pseudo_labels
import torch.nn as nn
import dataloaders
from types import SimpleNamespace
from models import *
from scipy.ndimage import label


class Trainer(BaseTrainer):
    def __init__(self, model, resume, config, supervised_loader, unsupervised_loader, iter_per_epoch, val_loader=None):
        super(Trainer, self).__init__(model, resume, config, iter_per_epoch)

        self.supervised_loader = supervised_loader
        self.unsupervised_loader = unsupervised_loader
        self.val_loader = val_loader

        self.num_classes = self.val_loader.dataset.num_classes
        # Get method parameter from FPA, FPA is initialized in train file
        self.method = self.model.module.method

        # TRANSORMS FOR VISUALIZATION
        self.restore_transform = transforms.Compose([
            DeNormalize(self.val_loader.MEAN, self.val_loader.STD),
            transforms.ToPILImage()])
        self.conf = config

        # self.pool_dims = config["normal_flow"]["pool_dims"]

        # self.NFdecoders = [load_decoder_arch(self.condition_vec, pool_dim) for pool_dim in self.pool_dims]
        # self.sub_epochs = config["normal_flow"]["sub_epochs"]
        self.nf_gamma = 0.0
        self.nf_theta = torch.nn.Sigmoid()
        self.nf_log_theta = torch.nn.LogSigmoid()
        self.start_time = time.time()
        self.nf_conf = SimpleNamespace(**self.config["nf_trainer"])
        self.nf_decoders = [
            load_decoder_arch(self.nf_conf, dim).to('cuda:0') for dim in self.nf_conf.pool_dims
        ]

        params = list(self.nf_decoders[0].parameters())
        for l in range(1, self.nf_conf.pool_layers):
            params += list(self.nf_decoders[l].parameters())
        self.nf_optimizer = torch.optim.Adam(params, lr=self.nf_conf.lr)

        self.nf_dataloader = dataloaders.CDDataset(self.conf['nf_train_supervised'])
        # self.nf_testloader = dataloaders.CDDataset(self.conf['nf_test_supervised'])

    def _first_stage_train(self, epoch):
        self.model.train()
        if self.method == 'supervised':
            dataloader = iter(self.supervised_loader)
            tbar = tqdm(range(len(self.supervised_loader)), ncols=160, position=0)
        else:
            dataloader = iter(zip(cycle(self.supervised_loader), self.unsupervised_loader))
            tbar = tqdm(range(len(self.unsupervised_loader)), ncols=160, position=0)

        self._reset_metrics()
        for batch_idx in tbar:
            if self.method == 'supervised':
                (A_l, B_l, target_l), (WA_ul, WB_ul, SA_ul, SB_ul, target_ul) = next(dataloader), (
                None, None, None, None, None)
            else:
                (A_l, B_l, target_l), (WA_ul, WB_ul, SA_ul, SB_ul, target_ul) = next(dataloader)
                WA_ul, WB_ul = WA_ul.cuda(non_blocking=True), WB_ul.cuda(non_blocking=True)
                SA_ul, SB_ul = SA_ul.cuda(non_blocking=True), SB_ul.cuda(non_blocking=True)
                target_ul = target_ul.cuda(non_blocking=True)
            A_l, B_l, target_l = A_l.cuda(non_blocking=True), B_l.cuda(non_blocking=True), target_l.cuda(
                non_blocking=True)
            self.optimizer.zero_grad()

            total_loss, cur_losses, outputs = self.model(epoch = epoch, A_l=A_l, B_l=B_l, target_l=target_l,
                                                         WA_ul=WA_ul, WB_ul=WB_ul, SA_ul=SA_ul, SB_ul=SB_ul)

            total_loss = total_loss.mean()
            total_loss.backward()
            self.optimizer.step()

            self._update_losses(cur_losses)
            self._compute_metrics(outputs, target_l, target_ul, epoch - 1)

            del A_l, B_l, target_l, WA_ul, WB_ul, SA_ul, SB_ul, target_ul
            del total_loss, cur_losses, outputs

            if self.method == 'supervised':
                tbar.set_description('Stage {} | Epoch ({}) | Ls {:.4f} Lu {:.4f} IoU(change-l) {:.3f}| '.format(
                    self.curr_stage, epoch, self.loss_l.average, self.loss_ul.average, self.class_iou_l[1]))
            else:
                tbar.set_description(
                    'T ({}) | Ls: {:.4f} Lu: {:.4f} IoU(change-l): {:.3f} IoU(change-ul): {:.3f} F1(ul): {:.3f} Kappa(ul): {:.3f}|'.format(
                        epoch, self.loss_l.average, self.loss_ul.average, self.class_iou_l[1], self.class_iou_ul[1],
                        self.f1_ul, self.kappa_ul))

            self.lr_scheduler.step(epoch=epoch - 1)

    def _second_stage_train(self, epoch, nf_decoders):
        nf_decoders = [decoder.train() for decoder in nf_decoders]
        self.model.eval()
        self.zdj = 1
        adjust_learning_rate(self.nf_conf, self.nf_optimizer, epoch)

        dataloader = iter(self.nf_dataloader if self.method == 'supervised' else zip(cycle(self.supervised_loader),
                                                                                     self.unsupervised_loader))
        I = len(self.nf_dataloader if self.method == 'supervised' else self.unsupervised_loader)
        P = self.nf_conf.condition_vec
        train_loss = 0.0
        train_count = 0
        self._reset_metrics()

        for batch_idx in tqdm(range(I)):
            if self.method == 'supervised':
                (A_l, B_l, target_l), (WA_ul, WB_ul, SA_ul, SB_ul, target_ul) = next(dataloader), (
                None, None, None, None, None)
            else:
                (A_l, B_l, target_l), (WA_ul, WB_ul, SA_ul, SB_ul, target_ul) = next(dataloader)
                WA_ul, WB_ul = WA_ul.cuda(non_blocking=True), WB_ul.cuda(non_blocking=True)
                SA_ul, SB_ul = SA_ul.cuda(non_blocking=True), SB_ul.cuda(non_blocking=True)
                target_ul = target_ul.cuda(non_blocking=True)

            A_l, B_l, target_l = A_l.cuda(non_blocking=True), B_l.cuda(non_blocking=True), target_l.cuda(
                non_blocking=True)
            _ = self.model(epoch=epoch,
                           A_l=A_l, B_l=B_l, target_l=target_l,
                           WA_ul=WA_ul, WB_ul=WB_ul, SA_ul=SA_ul, SB_ul=SB_ul)

            activations_list = activations.values()
            N = self.nf_conf.N
            lr = self.nf_optimizer.param_groups[0]['lr']

            for l, activation in enumerate(activations_list):
                e = activation.detach()
                B, C, H, W = e.size()
                S = H * W
                E = B * S  # E is the original size, total number of pixels before filtering
                p = positionalencoding2d(P, H, W).to('cuda:0').unsqueeze(0).repeat(B, 1, 1, 1)
                c_r = p.reshape(B, P, S).transpose(1, 2).reshape(E, P)  # BHWxP
                e_r = e.reshape(B, C, S).transpose(1, 2).reshape(E, C)  # BHWxC

                # Use target_l to generate mask, filter out pixels at position 0
                target_flat = target_l.view(-1)  # Flatten target_l to 1D
                mask = target_flat == 0  # Generate a boolean mask for positions with value 0

                # Select corresponding pixels in e_r based on mask
                e_p = e_r[mask]  # Only select pixels at positions where target_l is 0

                decoder = nf_decoders[l]

                # Update E to filtered size
                E_filtered = e_p.size(0)  # E_filtered is now the number of pixels after filtering
                FIB = E_filtered // N  # Recalculate FIB based on filtered E_filtered

                assert FIB > 0, 'MAKE SURE WE HAVE ENOUGH FIBERS, otherwise decrease N or batch-size!'

                for f in range(FIB):
                    idx = torch.arange(f * N, (f + 1) * N)
                    e_p_batch = e_p[idx]  # Get e_p for each batch
                    if 'cflow' in self.nf_conf.dec_arch:
                        z, log_jac_det = decoder(e_p_batch, [c_r, ])  # Pass c_r if needed
                    else:
                        z, log_jac_det = decoder(e_p_batch)  # c_r not needed

                    decoder_log_prob = get_logp(C, z, log_jac_det)
                    log_prob = decoder_log_prob / C
                    loss = -self.nf_log_theta(log_prob)
                    self.nf_optimizer.zero_grad()
                    loss.mean().backward()
                    self.nf_optimizer.step()
                    train_loss += t2np(loss.sum())
                    train_count += len(loss)

        mean_train_loss = train_loss / train_count if train_count > 0 else 0
        os.makedirs(Path(self.checkpoint_dir) / 'stage2', exist_ok=True)
        with open(Path(self.checkpoint_dir) / 'stage2' / "stage_2_training_log.txt", "a", encoding="utf-8") as file:
            file.write('Stage {} | Epoch: {:d} | train loss: {:.4f}, lr={:.6f}\n'
                       .format(self.curr_stage, epoch, mean_train_loss, lr))
        print('Stage {} | Epoch: {:d} \t train loss: {:.4f}, lr={:.6f}'
              .format(self.curr_stage, epoch, mean_train_loss, lr))

        self.improve_nf = False
        if mean_train_loss < self.min_nf_train_loss:
            self.improve_nf = True
            self.min_nf_train_loss = mean_train_loss
        self._nf_save_weights(epoch, save_best=self.improve_nf)
        # if epoch % self.config['trainer']['val_per_epochs_nf'] == 0:
        #     self.nf_mnt_curr = self._test_second_stage_train(epoch, nf_decoders)
        #     self.nf_improved = (self.nf_mnt_curr > self.nf_mnt_best)
        #     self.nf_mnt_best = self.nf_mnt_curr if self.nf_improved else self.nf_mnt_best
        # if epoch % self.config['trainer']['save_period_nf'] == 0:
        #     self._nf_save_weights(epoch, save_best=self.nf_improved)
    # Output id, save original image
    def _test_second_stage_train(self, epoch, nf_decoders):
        nf_decoders = [decoder.eval() for decoder in nf_decoders]

        self.model.eval()
        dataloader = iter(self.unsupervised_loader)
        tbar = tqdm(range(len(self.unsupervised_loader)), ncols=160, position=0)

        I = len(tbar)
        P = self.nf_conf.condition_vec
        
        total_inter = 0
        total_union = 0
        max_array = []
        min_array = []
        for batch_idx in tbar:
            (WA_ul, WB_ul, SA_ul, SB_ul, target_ul) = next(dataloader)
            WA_ul, WB_ul = WA_ul.cuda(non_blocking=True), WB_ul.cuda(non_blocking=True)
            SA_ul, SB_ul = SA_ul.cuda(non_blocking=True), SB_ul.cuda(non_blocking=True)
            target_ul = target_ul.cuda(non_blocking=True)


            # _, _, outputs
            _ = self.model(A_l=WA_ul, B_l=WB_ul, target_l=target_ul)

            activations_list = activations.values()
            test_dist = [list() for activation in activations_list]
            N = self.nf_conf.N
            for l, activation in enumerate(activations_list):
                e = activation.detach()  # BxCxHxW
                B, C, H, W = e.size()
                S = H * W
                E = B * S

                p = positionalencoding2d(P, H, W).to('cuda:0').unsqueeze(0).repeat(B, 1, 1, 1)
                c_r = p.reshape(B, P, S).transpose(1, 2).reshape(E, P)  # BHWxP
                e_r = e.reshape(B, C, S).transpose(1, 2).reshape(E, C)  # BHWxC
                decoder = nf_decoders[l]
                FIB = E // N + int(E % N > 0)  # number of fiber batches

                for f in range(FIB):
                    if f < (FIB - 1):
                        idx = torch.arange(f * N, (f + 1) * N)
                    else:
                        idx = torch.arange(f * N, E)
                    #
                    c_p = c_r[idx]  # NxP
                    e_p = e_r[idx]  # NxC

                    #
                    if 'cflow' in self.nf_conf.dec_arch:
                        z, log_jac_det = decoder(e_p, [c_p, ])
                    else:
                        z, log_jac_det = decoder(e_p)

                    decoder_log_prob = get_logp(C, z, log_jac_det)
                    log_prob = decoder_log_prob / C  # likelihood per dim

                    loss = -self.nf_log_theta(log_prob)
                    test_dist[l] = test_dist[l] + log_prob.detach().cpu().tolist()

            test_map = [list() for activation in activations_list]
            for l, activation in enumerate(activations_list):

                test_norm = torch.tensor(test_dist[l], dtype=torch.double)  # EHWx1

                test_norm -= torch.max(test_norm)  # normalize likelihoods to (-Inf:0] by subtracting a constant
                test_prob = torch.exp(test_norm)  # convert to probs in range [0:1]
                test_mask = test_prob.reshape(-1, activation.shape[2], activation.shape[3])
                test_mask = test_prob.reshape(-1, activation.shape[2], activation.shape[3])
                # upsample
                test_map[l] = F.interpolate(test_mask.unsqueeze(1),
                                            size=self.conf['nf_train_supervised']['crop_size'], mode='bilinear',
                                            align_corners=True).squeeze().numpy()
            # score aggregation
            score_map = np.zeros_like(test_map[0])
            for l, activation in enumerate(activations_list):
                score_map += test_map[l]
            
            
            score_mask = score_map
            # invert probs to anomaly scores
            super_mask = score_mask.max() - score_mask  # score_mask.max() gets a tensor feature map of the same size filled with max values

            # Binary classification to get final result gt_label
            nf_prop_map = torch.tensor(super_mask).to("cuda:0")
            max_array.append(nf_prop_map.max().item())
            min_array.append(nf_prop_map.min().item())
            if self.nf_conf.pool_layers == 1:
                fake_label = nf_prop_map.ge(0.9).long()
                thr_nf = 0.9
            elif self.nf_conf.pool_layers == 2:
                fake_label = nf_prop_map.ge(1.60).long()
                thr_nf = 1.60
            else:
                fake_label = nf_prop_map.ge(2.40).long()
                thr_nf = 2.40

            #inter, union = inter_over_union(fake_label.detach().cpu().numpy(), target_l.detach().cpu().numpy(), 2)
            inter, union = compute_intersection_union(fake_label.detach().cpu(), target_ul.detach().cpu())
            total_inter, total_union = total_inter + inter, total_union + union
            if batch_idx % 100 == 0:
                save_batch_prob_and_labels(WA_ul, WB_ul, nf_prop_map, target_ul, thr_nf,
                                        f'{self.checkpoint_dir}/stage2/nf_fake_label/test_batch{batch_idx}')
            del WA_ul, WB_ul, SA_ul, SB_ul, target_ul


        nf_iou_total = 1.0 * total_inter / (np.spacing(1) + total_union)
        with open(Path(self.checkpoint_dir) / 'stage2' / "stage_2_test_log.txt", "a", encoding="utf-8") as file:
            file.write('Stage2 IOU[0]: {}  IOU[1]: {}\n'.format(nf_iou_total[0], nf_iou_total[1]))
        # print(f"Stage {self.curr_stage} | valitated epoch {epoch} | nf iou (change): {(nf_iou_total)[1]}")
        return nf_iou_total[1]

    def _generate_pseudo_labels(self):
        self.model.eval()
        nf_decoders = [decoder.eval() for decoder in self.nf_decoders]
        log_theta = torch.nn.LogSigmoid()

        # Create storage directory
        fake_label_dir = self.conf["fake_labels_dir"]
        os.makedirs(fake_label_dir, exist_ok=True)

        # Iterate through labeled and unlabeled data simultaneously
        dataloader = iter(zip(cycle(self.supervised_loader), self.unsupervised_loader))
        total_batches = len(self.unsupervised_loader)
        tbar = tqdm(range(total_batches), ncols=160, position=0)

        for batch_idx in tbar:
            # Get current batch data
            (A_l, B_l, target_l), (WA_ul, WB_ul, SA_ul, SB_ul, target_ul) = next(dataloader)

            # Process labeled data
            self._process_single_batch(
                data=(A_l, B_l, target_l),
                batch_idx=batch_idx,
                prefix="Label",
                fake_label_dir=fake_label_dir,
                nf_decoders=nf_decoders
            )

            # Process unlabeled data (using weakly augmented data)
            self._process_single_batch(
                data=(WA_ul, WB_ul, target_ul),
                batch_idx=batch_idx,
                prefix="noLabel",
                fake_label_dir=fake_label_dir,
                nf_decoders=nf_decoders
            )

    def _process_single_batch(self, data, batch_idx, prefix, fake_label_dir, nf_decoders):
        A, B, target = data
        with torch.no_grad():
            # Transfer data to GPU
            A = A.cuda(non_blocking=True)
            B = B.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # Forward pass to get activations
            _ = self.model(A_l=A, B_l=B, target_l=target, method="supervised")
            activations_list = activations.values()  # Assume activations are obtained here

            # Initialize storage structure
            P = self.nf_conf.condition_vec
            N = self.nf_conf.N
            test_dist = [[] for _ in activations_list]
            crop_size = self.conf['nf_train_supervised']['crop_size']

            # Iterate through each feature layer
            for layer_idx, activation in enumerate(activations_list):
                e = activation.detach()
                B, C, H, W = e.shape
                S = H * W
                E = B * S

                # Create positional encoding
                pos_enc = positionalencoding2d(P, H, W).to('cuda')
                pos_enc = pos_enc.unsqueeze(0).repeat(B, 1, 1, 1)  # BxPxHxW

                # Reshape features and positional encoding
                pos_enc_flat = pos_enc.reshape(B, P, S).transpose(1, 2).reshape(E, P)  # ExP
                feat_flat = e.reshape(B, C, S).transpose(1, 2).reshape(E, C)  # ExC

                # Random permutation
                perm = torch.randperm(E)
                decoder = nf_decoders[layer_idx]
                FIB = E // N

                # Process in chunks
                for f in range(FIB + 1):
                    start_idx = f * N
                    end_idx = min((f + 1) * N, E) if f < FIB else E
                    if start_idx >= end_idx:
                        continue

                    # Get current chunk
                    idx = perm[start_idx:end_idx]
                    c_p = pos_enc_flat[idx]  # NxP
                    e_p = feat_flat[idx]  # NxC

                    # Pass through normalizing flow
                    if 'cflow' in self.nf_conf.dec_arch:
                        z, log_jac_det = decoder(e_p, [c_p])
                    else:
                        z, log_jac_det = decoder(e_p)

                    # Calculate probability
                    decoder_log_prob = get_logp(C, z, log_jac_det)
                    log_prob = decoder_log_prob / C  # Normalize per dimension
                    test_dist[layer_idx].extend(log_prob.cpu().tolist())

            # Generate probability map
            score_map = None
            for layer_idx, activation in enumerate(activations_list):
                # Normalize probabilities
                layer_probs = torch.tensor(test_dist[layer_idx], dtype=torch.double)
                layer_probs -= torch.max(layer_probs)
                probs = torch.exp(layer_probs)

                # Reshape to feature map size
                B, C, H, W = activation.shape
                probs = probs.reshape(B, H, W)

                # Upsample to original size
                upsampled = F.interpolate(
                    probs.unsqueeze(1),
                    size=crop_size,
                    mode='bilinear',
                    align_corners=True
                ).squeeze().numpy()

                # Accumulate scores
                if score_map is None:
                    score_map = np.zeros_like(upsampled)
                score_map += upsampled

            # Generate pseudo labels (invert probabilities)
            super_mask = score_map.max() - score_map
            pseudo_labels = torch.tensor(super_mask, device="cuda")

            # Save results
            filename = f"{prefix}_batch_{batch_idx}.pt"
            save_path = os.path.join(fake_label_dir, filename)
            print(f"Saving pseudo labels to {save_path}")
            torch.save(pseudo_labels.cpu(), save_path)

    def _third_stage_train(self, epoch):
        cnt_all = 0
        self.model.train()
        nf_decoders = [decoder.train() for decoder in self.nf_decoders]
        theta = torch.nn.Sigmoid()
        log_theta = torch.nn.LogSigmoid()

        dataloader = iter(zip(cycle(self.supervised_loader), self.unsupervised_loader))
        tbar = tqdm(range(len(self.unsupervised_loader)), ncols=160, position=0)

        I = len(tbar)
        P = self.nf_conf.condition_vec
        self._reset_metrics()

        # Pseudo label storage path (consistent with generation function)
        # fake_label_dir = os.path.join(self.checkpoint_dir, "fake_labels")
        fake_label_dir = self.config["fake_labels_dir"]
        os.makedirs(fake_label_dir, exist_ok=True)

        for batch_idx in tbar:
            # Load data and pseudo labels ================================
            (A_l, B_l, target_l), (WA_ul, WB_ul, SA_ul, SB_ul, target_ul) = next(dataloader)

            # Load pseudo labels for labeled data
            label_path = os.path.join(fake_label_dir, f"Label_batch_{batch_idx}.pt")
            nf_l_prop_map = torch.load(label_path).to("cuda")

            # Load pseudo labels for unlabeled data
            nolabel_path = os.path.join(fake_label_dir, f"noLabel_batch_{batch_idx}.pt")
            nf_ul_prop_map = torch.load(nolabel_path).to("cuda")
            # ==============================================

            # Transfer data to GPU
            WA_ul, WB_ul = WA_ul.cuda(non_blocking=True), WB_ul.cuda(non_blocking=True)
            SA_ul, SB_ul = SA_ul.cuda(non_blocking=True), SB_ul.cuda(non_blocking=True)
            target_ul = target_ul.cuda(non_blocking=True)
            A_l, B_l, target_l = A_l.cuda(non_blocking=True), B_l.cuda(non_blocking=True), target_l.cuda(
                non_blocking=True)

            self.optimizer.zero_grad()

            # Model forward pass (pass in dual pseudo labels) =======================
            vis_savedir = os.path.join(self.checkpoint_dir, "visual")
            total_third_step_loss, cur_third_step_losses, outputs= self.model(
                epoch=epoch,
                idx=batch_idx,
                vis_savedir=vis_savedir,
                A_l=A_l,
                B_l=B_l,
                target_l=target_l,
                WA_ul=WA_ul,
                WB_ul=WB_ul,
                SA_ul=SA_ul,
                SB_ul=SB_ul,
                target_ul_real=target_ul,
                nf_l=nf_l_prop_map,  # New labeled pseudo label
                nf_ul=nf_ul_prop_map,  # Original unlabeled pseudo label
                method="third"
            )
            # ==============================================

            # cnt_all += cnt
            total_third_step_loss = total_third_step_loss.mean()
            total_third_step_loss.backward()
            self.optimizer.step()

            self._update_losses(cur_third_step_losses)
            self._compute_metrics(outputs, target_l, target_ul, epoch - 1)

            # Memory cleanup
            del A_l, B_l, target_l, WA_ul, WB_ul, SA_ul, SB_ul, target_ul
            del total_third_step_loss, cur_third_step_losses, outputs

            # Progress bar display
            tbar.set_description(
                'T ({}) | Ls: {:.4f} Lu: {:.4f} IoU(change-l): {:.3f} IoU(change-ul): {:.3f} F1(ul): {:.3f} Kappa(ul): {:.3f}|\n'.format(
                    epoch, self.loss_l.average, self.loss_ul.average, self.class_iou_l[1], self.class_iou_ul[1],
                    self.f1_ul, self.kappa_ul))

            # Log recording
            # with open(Path(self.checkpoint_dir) / "stage_3_training_log.txt", "a", encoding="utf-8") as file:
            #     file.write(
            #         'T ({}) | Ls: {:.4f} Lu: {:.4f} IoU(change-l): {:.3f} IoU(change-ul): {:.3f} F1(ul): {:.3f} Kappa(ul): {:.3f}|\n'.format(
            #             epoch, self.loss_l.average, self.loss_ul.average, self.class_iou_l[1], self.class_iou_ul[1],
            #             self.f1_ul, self.kappa_ul))

            self.lr_scheduler.step(epoch=epoch - 1)

        # Final log recording
        # with open(Path(self.checkpoint_dir) / "stage_3_training_log.txt", "a", encoding="utf-8") as file:
        #     file.write('Total_cnt: {}\n'.format(cnt_all))
        

    def _train_epoch(self, epoch, stage='one', nf_decoders=None):
        if stage == 'one':
            self._first_stage_train(epoch)
        elif stage == 'two':
            self._second_stage_train(epoch, nf_decoders)
        elif stage == 'three':
            self._third_stage_train(epoch)
        else:
            raise ValueError(f"Stage '{stage}' is not implemented.")

    def _valid_epoch(self, epoch):
        print('###### EVALUATION ######')

        self.model.eval()
        total_loss_val = AverageMeter()
        total_inter, total_union = 0, 0
        total_tp, total_fp, total_tn, total_fn = 0, 0, 0, 0

        tbar = tqdm(self.val_loader, ncols=150)
        l=len(tbar)
        with torch.no_grad():
            for batch_idx, (A, B, target, _) in enumerate(tbar):
                target, A, B = target.cuda(non_blocking=True), A.cuda(non_blocking=True), B.cuda(non_blocking=True)

                H, W = target.size(1), target.size(2)
                up_sizes = (ceil(H / 8) * 8, ceil(W / 8) * 8)
                pad_h, pad_w = up_sizes[0] - A.size(2), up_sizes[1] - A.size(3)
                A = F.pad(A, pad=(0, pad_w, 0, pad_h), mode='reflect')
                B = F.pad(B, pad=(0, pad_w, 0, pad_h), mode='reflect')
                output = self.model(A_l=A, B_l=B)
                output = output[:, :, :H, :W]

                # LOSS
                loss = F.cross_entropy(output, target)
                total_loss_val.update(loss.item())

                correct, labeled, inter, union, tp, fp, tn, fn = eval_metrics(output, target, self.num_classes)
                total_inter, total_union = total_inter + inter, total_union + union
                total_tp, total_fp = total_tp + tp, total_fp + fp
                total_tn, total_fn = total_tn + tn, total_fn + fn

                # PRINT INFO
                IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
                P = 1.0 * total_tp / (total_tp + total_fp + np.spacing(1))
                R = 1.0 * total_tp / (total_tp + total_fn + np.spacing(1))
                F1 = 2 * P * R / (P + R + np.spacing(1))
                OA = (total_tp + total_tn) / (total_tp + total_fp + total_tn + total_fn + np.spacing(1))
                PRE = (total_tp + total_fn) * (total_tp + total_fp) / (
                            (total_tp + total_fp + total_tn + total_fn + np.spacing(1)) ** 2) \
                      + (total_tn + total_fp) * (total_tn + total_fn) / (
                                  (total_tp + total_fp + total_tn + total_fn + np.spacing(1)) ** 2)
                Kappa = (OA - PRE) / (1 - PRE)

                seg_metrics = {"Class_IoU": dict(zip(range(self.num_classes), np.round(IoU, 3)))}

                tbar.set_description('Stage {} | EVAL ({}) | Loss: {:.3f}, IoU(change): {:.4f}, F1: {:.4f} , Kappa: {:.4f} |'. \
                                     format(self.curr_stage, epoch, total_loss_val.average, IoU[1], F1, Kappa))
                if batch_idx==l-1:
                    if self.curr_stage == 'three':
                        with open(Path(self.checkpoint_dir) /"val_log.txt", "a", encoding="utf-8") as file:
                            file.write('Stage {} | EVAL ({}) | Loss: {:.3f}, IoU(change): {:.4f}, F1: {:.4f} , Kappa: {:.4f} |\n'. \
                                             format(self.curr_stage, epoch, total_loss_val.average, IoU[1], F1, Kappa))
                    elif self.curr_stage == 'one':
                        with open(Path(self.checkpoint_dir) /'stage1'/"val_log.txt", "a", encoding="utf-8") as file:
                            file.write('Stage {} | EVAL ({}) | Loss: {:.3f}, IoU(change): {:.4f}, F1: {:.4f} , Kappa: {:.4f} |\n'. \
                                             format(self.curr_stage, epoch, total_loss_val.average, IoU[1], F1, Kappa))

            if (time.time() - self.start_time) / 3600 > 22:
                self._save_checkpoint(epoch, stage="stage-three", save_best=self.improved)

        return IoU[1]

    def _proportion_val(self):
        cnt=0
        global_records = []
        dataloader = iter(zip(cycle(self.supervised_loader), self.unsupervised_loader))
        tbar = tqdm(range(len(self.unsupervised_loader)), ncols=160, position=0)

        I = len(tbar)
        # Ensure directory for storing fake_label exists
        fake_label_dir = os.path.join(self.checkpoint_dir, "fake_labels")
        os.makedirs(fake_label_dir, exist_ok=True)
        # if self.fake_label:
        #     fake_label_dir = self.config["trainer"]["fake_label_dir"]

        for batch_idx in tbar:
            (A_l, B_l, target_l), (WA_ul, WB_ul, SA_ul, SB_ul, target_ul) = next(dataloader)
            WA_ul, WB_ul = WA_ul.cuda(non_blocking=True), WB_ul.cuda(non_blocking=True)
            SA_ul, SB_ul = SA_ul.cuda(non_blocking=True), SB_ul.cuda(non_blocking=True)
            target_ul = target_ul.cuda(non_blocking=True)
            A_l, B_l, target_l = A_l.cuda(non_blocking=True), B_l.cuda(non_blocking=True), target_l.cuda(
                non_blocking=True
            )
            # fake_label file path
            fake_label_path = os.path.join(fake_label_dir, f"fake_label_batch_{batch_idx}.pt")
            # Load fake_label from disk
            nf_prop_map = torch.load(fake_label_path).to("cuda")

            self._view_val_proportion(nf_prop_map, target_ul, global_records)
            # cnt +=1
            # if cnt == 10:
            #     break
        self._visualize_results(global_records)

    def _view_val_proportion(self, nf_prop_map, target_ul, global_records):
        # Generate threshold list [0.3, 0.6, ..., 2.7]
        thresholds = np.arange(0.3, 3.0, 0.3)

        # Iterate through each channel
        for channel in range(target_ul.shape[0]):
            # Get current channel data
            current_target = target_ul[channel]
            current_prop = nf_prop_map[channel]

            # Find connected regions
            current_target_np = current_target.cpu().numpy()
            labeled_array, num_features = label(current_target_np == 1)

            # Iterate through each connected component
            for i in range(1, num_features + 1):
                # Get current connected component mask
                mask = (labeled_array == i)

                # Get corresponding prop region
                prop_values = current_prop[mask]

                # Calculate proportions for different thresholds
                proportions = {}
                total_pixels = mask.sum()

                for thresh in thresholds:
                    count = (prop_values > thresh).sum()
                    proportions[round(thresh, 1)] = count / total_pixels

                # Save results
                global_records.append({
                    'channel': channel,
                    'block_id': i,
                    'proportions': proportions
                })

    # Visualization function
    def _visualize_results(self, global_records):
        plt.figure(figsize=(10, 6))

        # Iterate through each block to draw a curve
        for record in global_records:
            # Get proportion dictionary for current block
            proportions = record['proportions']
            # Sort thresholds (keys are float type)
            thresholds = sorted(proportions.keys())
            y_values = []
            for thresh in thresholds:
                value = proportions[thresh]
                # If value is tensor, convert to CPU and Python numeric value
                if isinstance(value, torch.Tensor):
                    value = value.cpu().item()
                y_values.append(value)

            # Optional: add label for each curve showing channel and block id
            label_str = f"Batch {record['channel']} Block {record['block_id']}"
            plt.plot(thresholds, y_values, marker='o')

        plt.xlabel('Threshold Value')
        plt.ylabel('Proportion')
        plt.title('Pixel Proportion vs Threshold')
        plt.grid(True)
        plt.legend()  # Display legend
        plt.savefig('gt_proportion_val.png', dpi=300)

    def _components_tongji(self, nf_decoders):
        nf_decoders = [decoder.eval() for decoder in nf_decoders]

        self.model.eval()
        dataloader = iter(self.unsupervised_loader)
        tbar = tqdm(range(len(self.unsupervised_loader)), ncols=160, position=0)

        I = len(tbar)
        P = self.nf_conf.condition_vec

        all_val = 0
        all_cnt = 0
        self.val_list = []
        for batch_idx in tbar:
            (WA_ul, WB_ul, SA_ul, SB_ul, target_ul) = next(dataloader)
            WA_ul, WB_ul = WA_ul.cuda(non_blocking=True), WB_ul.cuda(non_blocking=True)
            SA_ul, SB_ul = SA_ul.cuda(non_blocking=True), SB_ul.cuda(non_blocking=True)
            target_ul = target_ul.cuda(non_blocking=True)

            # _, _, outputs
            _ = self.model(A_l=WA_ul, B_l=WB_ul, target_l=target_ul)

            activations_list = activations.values()
            test_dist = [list() for activation in activations_list]
            N = self.nf_conf.N
            for l, activation in enumerate(activations_list):
                e = activation.detach()  # BxCxHxW
                B, C, H, W = e.size()
                S = H * W
                E = B * S

                p = positionalencoding2d(P, H, W).to('cuda').unsqueeze(0).repeat(B, 1, 1, 1)
                c_r = p.reshape(B, P, S).transpose(1, 2).reshape(E, P)  # BHWxP
                e_r = e.reshape(B, C, S).transpose(1, 2).reshape(E, C)  # BHWxC
                decoder = nf_decoders[l]
                FIB = E // N + int(E % N > 0)  # number of fiber batches

                for f in range(FIB):
                    if f < (FIB - 1):
                        idx = torch.arange(f * N, (f + 1) * N)
                    else:
                        idx = torch.arange(f * N, E)
                    #
                    c_p = c_r[idx]  # NxP
                    e_p = e_r[idx]  # NxC

                    #
                    if 'cflow' in self.nf_conf.dec_arch:
                        z, log_jac_det = decoder(e_p, [c_p, ])
                    else:
                        z, log_jac_det = decoder(e_p)

                    decoder_log_prob = get_logp(C, z, log_jac_det)
                    log_prob = decoder_log_prob / C  # likelihood per dim

                    loss = -self.nf_log_theta(log_prob)
                    test_dist[l] = test_dist[l] + log_prob.detach().cpu().tolist()

            test_map = [list() for activation in activations_list]
            for l, activation in enumerate(activations_list):
                test_norm = torch.tensor(test_dist[l], dtype=torch.double)  # EHWx1

                test_norm -= torch.max(test_norm)  # normalize likelihoods to (-Inf:0] by subtracting a constant
                test_prob = torch.exp(test_norm)  # convert to probs in range [0:1]
                test_mask = test_prob.reshape(-1, activation.shape[2], activation.shape[3])
                test_mask = test_prob.reshape(-1, activation.shape[2], activation.shape[3])
                # upsample
                test_map[l] = F.interpolate(test_mask.unsqueeze(1),
                                            size=self.conf['nf_train_supervised']['crop_size'], mode='bilinear',
                                            align_corners=True).squeeze().numpy()
            # score aggregation
            score_map = np.zeros_like(test_map[0])
            for l, activation in enumerate(activations_list):
                score_map += test_map[l]

            score_mask = score_map
            # invert probs to anomaly scores
            super_mask = score_mask.max() - score_mask  # score_mask.max() gets a tensor feature map of the same size filled with max values

            # Binary classification to get final result gt_label
            nf_prop_map = torch.tensor(super_mask).to("cuda")
            val, cnt = self._cal_val_cnt(nf_prop_map, target_ul)
            all_val += val
            all_cnt += cnt
            with open(Path(self.checkpoint_dir) / 'stage2' / "components_tongji.txt", "a", encoding="utf-8") as file:
                file.write(f"{all_val / all_cnt:.6f}\n")
            del WA_ul, WB_ul, SA_ul, SB_ul, target_ul

        plt.plot(self.val_list)
        plt.title("Line Plot of self.val_list")
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.savefig(Path(self.checkpoint_dir) / 'stage2' / "tongji.jpg")
        print(max(self.val_list))
        print(min(self.val_list))

    def _cal_val_cnt(self, nf_prop_map, target_ul):

        val = 0
        cnt = 0
        # Iterate through each sample
        for b in range(target_ul.shape[0]):
            # Convert to numpy array on CPU for connected region analysis
            target_np = target_ul[b].cpu().numpy().astype(np.uint8)
            prop_np = nf_prop_map[b].cpu().numpy()

            # Extract connected regions with value 1
            labeled_mask, num_components = label(target_np)
            cnt += num_components
            # Iterate through each connected component
            for c in range(1, num_components + 1):
                # Get mask for current connected component
                component_mask = (labeled_mask == c)

                # Calculate average confidence for this connected component
                conf_values = prop_np[component_mask]
                avg_conf = conf_values.mean() if len(conf_values) > 0 else 0.0
                val += avg_conf
                self.val_list.append(avg_conf)
        return val, cnt
    def _reset_metrics(self):
        self.loss_l = AverageMeter()
        self.loss_ul = AverageMeter()
        self.loss_ul_alg = AverageMeter()
        self.loss_ul_cls = AverageMeter()
        self.loss_ul_nf = AverageMeter()
        self.total_inter_l, self.total_union_l = 0, 0
        self.total_correct_l, self.total_label_l = 0, 0
        self.total_inter_ul, self.total_union_ul = 0, 0
        self.total_correct_ul, self.total_label_ul = 0, 0
        self.total_tp_l, self.total_fp_l = 0, 0
        self.total_tp_ul, self.total_fp_ul = 0, 0
        self.total_tn_l, self.total_fn_l = 0, 0
        self.total_tn_ul, self.total_fn_ul = 0, 0
        self.pixel_acc_l, self.pixel_acc_ul = 0, 0
        self.class_iou_l, self.class_iou_ul = {}, {}
        self.f1_l, self.f1_ul = 0, 0
        self.kappa_l, self.kappa_ul = 0, 0

    def _update_losses(self, cur_losses):
        if "loss_l" in cur_losses.keys():
            self.loss_l.update(cur_losses['loss_l'].mean().item())
        if "loss_ul" in cur_losses.keys():
            self.loss_ul.update(cur_losses['loss_ul'].mean().item())
        if "loss_ul_alg" in cur_losses.keys():
            self.loss_ul_alg.update(cur_losses['loss_ul_alg'].mean().item())
        if "loss_ul_cls" in cur_losses.keys():
            self.loss_ul_cls.update(cur_losses['loss_ul_cls'].mean().item())
        if "loss_ul_nf" in cur_losses.keys():
            self.loss_ul_nf.update(cur_losses['loss_ul_nf'].mean().item())

    def _compute_metrics(self, outputs, target_l, target_ul, epoch):
        seg_metrics_l = eval_metrics(outputs['pred_l'], target_l, self.num_classes)
        self._update_seg_metrics(*seg_metrics_l, True)
        seg_metrics_l = self._get_seg_metrics(True)
        self.pixel_acc_l, self.class_iou_l, self.f1_l, self.kappa_l = seg_metrics_l.values()

        if self.method != 'supervised':
            
            seg_metrics_ul = eval_metrics(outputs['pred_ul'], target_ul, self.num_classes)
            self._update_seg_metrics(*seg_metrics_ul, False)
            seg_metrics_ul = self._get_seg_metrics(False)
            self.pixel_acc_ul, self.class_iou_ul, self.f1_ul, self.kappa_ul = seg_metrics_ul.values()

    def _update_seg_metrics(self, correct, labeled, inter, union, tp, fp, tn, fn, supervised=True):
        if supervised:
            self.total_correct_l += correct
            self.total_label_l += labeled
            self.total_inter_l += inter
            self.total_union_l += union
            self.total_tp_l += tp
            self.total_fp_l += fp
            self.total_tn_l += tn
            self.total_fn_l += fn
        else:
            self.total_correct_ul += correct
            self.total_label_ul += labeled
            self.total_inter_ul += inter
            self.total_union_ul += union
            self.total_tp_ul += tp
            self.total_fp_ul += fp
            self.total_tn_ul += tn
            self.total_fn_ul += fn

    def _get_seg_metrics(self, supervised=True):
        if supervised:
            pixAcc = 1.0 * self.total_correct_l / (np.spacing(1) + self.total_label_l)
            IoU = 1.0 * self.total_inter_l / (np.spacing(1) + self.total_union_l)
            P = 1.0 * self.total_tp_l / (self.total_tp_l + self.total_fp_l + np.spacing(1))
            R = 1.0 * self.total_tp_l / (self.total_tp_l + self.total_fn_l + np.spacing(1))
            F1 = 2 * P * R / (P + R + np.spacing(1))
            OA = (self.total_tp_l + self.total_tn_l) / (
                        self.total_tp_l + self.total_fp_l + self.total_tn_l + self.total_fn_l + np.spacing(1))
            PRE = (self.total_tp_l + self.total_fn_l) * (self.total_tp_l + self.total_fp_l) / (
                        (self.total_tp_l + self.total_fp_l + self.total_tn_l + self.total_fn_l + np.spacing(1)) ** 2) \
                  + (self.total_tn_l + self.total_fp_l) * (self.total_tn_l + self.total_fn_l) / ((
                                                                                                             self.total_tp_l + self.total_fp_l + self.total_tn_l + self.total_fn_l + np.spacing(
                                                                                                         1)) ** 2)
            Kappa = (OA - PRE) / (1 - PRE + np.spacing(1) ** 2)

        else:
            pixAcc = 1.0 * self.total_correct_ul / (np.spacing(1) + self.total_label_ul)
            IoU = 1.0 * self.total_inter_ul / (np.spacing(1) + self.total_union_ul)
            P = 1.0 * self.total_tp_ul / (self.total_tp_ul + self.total_fp_ul + np.spacing(1))
            R = 1.0 * self.total_tp_ul / (self.total_tp_ul + self.total_fn_ul + np.spacing(1))
            F1 = 2 * P * R / (P + R + np.spacing(1))
            OA = (self.total_tp_ul + self.total_tn_ul) / (
                        self.total_tp_ul + self.total_fp_ul + self.total_tn_ul + self.total_fn_ul + np.spacing(1))
            PRE = (self.total_tp_ul + self.total_fn_ul) * (self.total_tp_ul + self.total_fp_ul) / ((
                                                                                                               self.total_tp_ul + self.total_fp_ul + self.total_tn_ul + self.total_fn_ul + np.spacing(
                                                                                                           1)) ** 2) \
                  + (self.total_tn_ul + self.total_fp_ul) * (self.total_tn_ul + self.total_fn_ul) / ((
                                                                                                                 self.total_tp_ul + self.total_fp_ul + self.total_tn_ul + self.total_fn_ul + np.spacing(
                                                                                                             1)) ** 2)
            Kappa = (OA - PRE) / (1 - PRE + np.spacing(1) ** 2)

        return {
            "Pixel_Accuracy": np.round(pixAcc, 3),
            "Class_IoU": dict(zip(range(self.num_classes), np.round(IoU, 3))),
            "F1": np.round(F1, 4),
            "Kappa": np.round(Kappa, 4)
        }

    def _log_values(self, cur_losses):
        logs = {}
        if "loss_sup" in cur_losses.keys():
            logs['loss_sup'] = self.loss_sup.average
        if "loss_unsup" in cur_losses.keys():
            logs['loss_unsup'] = self.loss_unsup.average
        if "loss_weakly" in cur_losses.keys():
            logs['loss_weakly'] = self.loss_weakly.average
        if "pair_wise" in cur_losses.keys():
            logs['pair_wise'] = self.pair_wise.average

        logs['mIoU_labeled'] = self.mIoU_l
        logs['pixel_acc_labeled'] = self.pixel_acc_l
        if self.mode == 'semi':
            logs['mIoU_unlabeled'] = self.mIoU_ul
            logs['pixel_acc_unlabeled'] = self.pixel_acc_ul
        return logs

    def _write_scalars_tb(self, logs):
        for k, v in logs.items():
            if 'class_iou' not in k: self.writer.add_scalar(f'train/{k}', v, self.wrt_step)
        for i, opt_group in enumerate(self.optimizer.param_groups):
            self.writer.add_scalar(f'train/Learning_rate_{i}', opt_group['lr'], self.wrt_step)
        current_rampup = self.model.module.unsup_loss_w.current_rampup
        self.writer.add_scalar('train/Unsupervised_rampup', current_rampup, self.wrt_step)

    def _add_img_tb(self, val_visual, wrt_mode):
        val_img = []
        palette = self.val_loader.dataset.palette
        for imgs in val_visual:
            imgs = [self.restore_transform(i) if (isinstance(i, torch.Tensor) and len(i.shape) == 3)
                    else colorize_mask(i, palette) for i in imgs]
            imgs = [i.convert('RGB') for i in imgs]
            imgs = [self.viz_transform(i) for i in imgs]
            val_img.extend(imgs)
        val_img = torch.stack(val_img, 0)
        val_img = make_grid(val_img.cpu(), nrow=val_img.size(0) // len(val_visual), padding=5)
        self.writer.add_image(f'{wrt_mode}/inputs_targets_predictions', val_img, self.wrt_step)

    def _write_img_tb(self, A_l, B_l, target_l, A_ul, B_ul, target_ul, outputs, epoch):
        outputs_l_np = outputs['sup_pred'].data.max(1)[1].cpu().numpy()
        targets_l_np = target_l.data.cpu().numpy()
        imgs = [[i.data.cpu(), j.data.cpu(), k, l] for i, j, k, l in zip(A_l, B_l, outputs_l_np, targets_l_np)]
        self._add_img_tb(imgs, 'supervised')

        if self.mode == 'semi':
            outputs_ul_np = outputs['unsup_pred'].data.max(1)[1].cpu().numpy()
            targets_ul_np = target_ul.data.cpu().numpy()
            imgs = [[i.data.cpu(), j.data.cpu(), k, l] for i, j, k, l in zip(A_ul, B_ul, outputs_ul_np, targets_ul_np)]
            self._add_img_tb(imgs, 'unsupervised')
