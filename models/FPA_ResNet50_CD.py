import math, time
import random
from itertools import chain
import torch
import torch.nn.functional as F
from torch import nn
from base import BaseModel
from utils.helpers import *
from utils.losses import *
from models.decoder import *
from models.encoder import *
import torch
import numpy as np
from scipy.ndimage import label

class FPA_ResNet50_CD(BaseModel):
    def __init__(self, num_classes, config, loss_l=None, loss_alg=None, len_unsper=None, testing=False, pretrained=True):
        self.num_classes = num_classes
        if not testing:
            assert (loss_l is not None) 

        super(FPA_ResNet50_CD, self).__init__()
        self.config = config
        conf=config['model']
        self.method = conf['method']

        # Supervised and unsupervised losses        
        self.loss_l         = loss_l
        self.loss_alg       = loss_alg

        # confidence masking (sup mat)
        # if self.method != 'supervised':
        self.confidence_thr     = conf['confidence_thr']
        print ('thr: ', self.confidence_thr)

        # Create the model
        self.encoder = Encoder_ResNet50(pretrained=pretrained)

        # The main encoder
        upscale             = 8
        num_out_ch          = 2048
        decoder_in_ch       = num_out_ch // 4
        self.decoder        = Decoder(upscale, decoder_in_ch, num_classes=num_classes)

    def forward(self, epoch = None, A_l=None, B_l=None, target_l=None, \
                    WA_ul=None, WB_ul=None, SA_ul=None, SB_ul=None, target_ul=None, nf_ul=None, method=None):
        if not self.training:
            return self.decoder(self.encoder(A_l, B_l))
        input_size  = (A_l.size(2), A_l.size(3))                

        if method != None:
            self.method = method        

        # If supervised mode only, return
        if self.method == 'supervised':
            # Supervised loss
            out_l  = self.decoder(self.encoder(A_l, B_l))
            loss_l = self.loss_l(out_l, target_l) 
            curr_losses = {'loss_l': loss_l}
            total_loss = loss_l

            if out_l.shape != A_l.shape:
                out_l = F.interpolate(out_l, size=input_size, mode='bilinear', align_corners=True)
            outs = {'pred_l': out_l}
     
     
            return total_loss, curr_losses, outs
        
        elif self.method == "third":
            out_l  = self.decoder(self.encoder(A_l, B_l))
            loss_l = self.loss_l(out_l, target_l) 

            # Get main prediction
            weak_out_ul    = self.decoder(self.encoder(WA_ul, WB_ul))
            strong_feat_ul = self.encoder(SA_ul, SB_ul)
            strong_out_ul  = self.decoder(strong_feat_ul)

            # Generate pseudo_label
            weak_prob_ul = F.softmax(weak_out_ul.detach_(), dim=1)            
            max_probs, target_ul = torch.max(weak_prob_ul, dim=1)
            mask = max_probs.ge(self.confidence_thr).float()
            loss_ul_cls  = (F.cross_entropy(strong_out_ul, target_ul, reduction='none') * mask).mean()
            loss_ul_alg = self.loss_alg(weak_prob_ul, strong_feat_ul, self.confidence_thr)
            loss_ul = loss_ul_cls + loss_ul_alg

            # record loss
            curr_losses = {'loss_l': loss_l}
            curr_losses['loss_ul'] = loss_ul
            curr_losses['loss_ul_cls'] = loss_ul_cls
            curr_losses['loss_ul_alg'] = loss_ul_alg

            if weak_out_ul.shape != WA_ul.shape:
                out_l = F.interpolate(out_l, size=input_size, mode='bilinear', align_corners=True)
                weak_out_ul = F.interpolate(weak_out_ul, size=input_size, mode='bilinear', align_corners=True)
            outs = {'pred_l': out_l, 'pred_ul': weak_out_ul}

            # Compute the unsupervised loss
            total_loss  = loss_l + loss_ul  
            return total_loss, curr_losses, outs
        # If semi supervised mode
        else:
            # Supervised loss
            out_l  = self.decoder(self.encoder(A_l, B_l))
            loss_l = self.loss_l(out_l, target_l) 

            # Get main prediction
            weak_feat_ul   = self.encoder(WA_ul, WB_ul)
            weak_out_ul    = self.decoder(weak_feat_ul)
            strong_feat_ul = self.encoder(SA_ul, SB_ul)
            strong_out_ul  = self.decoder(strong_feat_ul)

            
                    
            # Generate pseudo_label
            weak_prob_ul = F.softmax(weak_out_ul.detach_(), dim=1)
            max_probs, target_ul = torch.max(weak_prob_ul, dim=1)
            mask = max_probs.ge(self.confidence_thr).float()
            loss_ul_cls  = (F.cross_entropy(strong_out_ul, target_ul, reduction='none') * mask).mean()  #PA_loss
            loss_ul_alg = self.loss_alg(weak_prob_ul, strong_feat_ul, self.confidence_thr)  
            loss_ul = loss_ul_cls + loss_ul_alg

            # record loss
            curr_losses = {'loss_l': loss_l}
            curr_losses['loss_ul'] = loss_ul
            curr_losses['loss_ul_cls'] = loss_ul_cls
            curr_losses['loss_ul_alg'] = loss_ul_alg

            if weak_out_ul.shape != WA_ul.shape:
                out_l = F.interpolate(out_l, size=input_size, mode='bilinear', align_corners=True)
                weak_out_ul = F.interpolate(weak_out_ul, size=input_size, mode='bilinear', align_corners=True)
            outs = {'pred_l': out_l, 'pred_ul': weak_out_ul}

            # Compute the unsupervised loss
            total_loss  = loss_l + loss_ul  
            
            return total_loss, curr_losses, outs


    def improve_pseudo_label(self, nf_prop_map, target_ul):
        """
        Optimize pseudo labels, filtering low-confidence regions based on average confidence of connected components.

        Parameters:
            nf_prop_map (Tensor): Probability map, size [B, H, W], values 0~1
            target_ul (Tensor): Original pseudo labels, size [B, H, W], connected components with values 0 or 1
            threshold (float): Confidence threshold, connected components below this value will be set to 1

        Returns:
            improve_target_ul (Tensor): Optimized pseudo labels, same size as target_ul
        """

        threshold=self.config["model"]["nf_thirdStageTrain"]
        device = target_ul.device
        improve_target_ul = target_ul.clone()  # Initialize output

        # Iterate through each sample
        for b in range(target_ul.shape[0]):
            # Convert to numpy array on CPU for connected region analysis
            target_np = target_ul[b].cpu().numpy().astype(np.uint8)
            prop_np = nf_prop_map[b].cpu().numpy()

            # Extract connected regions with value 1
            labeled_mask, num_components = label(target_np)

            # Iterate through each connected component
            for c in range(1, num_components + 1):
                # Get mask for current connected component
                component_mask = (labeled_mask == c)

                # Calculate average confidence for this connected component
                conf_values = prop_np[component_mask]
                avg_conf = conf_values.mean() if len(conf_values) > 0 else 0.0

                # If average confidence is below threshold, set this region to 0 in output
                if avg_conf < threshold:
                    # Convert mask to Tensor and transfer back to original device
                    component_tensor = torch.from_numpy(component_mask).to(device)
                    improve_target_ul[b][component_tensor] = 0

        return improve_target_ul

    def get_backbone_params(self):
        return self.encoder.get_backbone_params()

    def get_other_params(self):
        return chain(self.encoder.get_module_params(), self.decoder.parameters())
