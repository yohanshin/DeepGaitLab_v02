import os

import torch
import wandb
import psutil
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from configs import constants as _C

# Heatmap-soft argmax
from ..base_trainer import BaseTrainer
from ...models.backbone.vit import ViT
from ...models.losses import JointsMSELoss, JointGNLLLoss
from ...models.head.heatmap_head import TopdownHeatmapSimpleHead
from ...models.head.visibility_head import VisibilityPerLandmark
from ...models.detectors.utils.visualization import compare_results_denseldmks2d
from ...datasets.utils.heatmap_utils import keypoints_from_heatmap_soft_argmax

class DenseKPDetector(BaseTrainer):
    def __init__(self, cfg, viz_dir=None):
        super(DenseKPDetector, self).__init__(cfg)
        self.image_size = cfg.image_size

        backbone_cfg = {k: v for k, v in cfg.backbone.items() if not k in ['type', 'name']}
        self.backbone_cfg = backbone_cfg
        self.backbone = ViT(**backbone_cfg).to(self.device)
        
        decoder_cfg = {k: v for k, v in cfg.model['decoder'].items() if k != 'layer_name'}
        self.decoder = TopdownHeatmapSimpleHead(**decoder_cfg)
        self.hm_criteria = JointsMSELoss(True, cfg.loss_weights["heatmap"])
        self.joints_criteria = JointGNLLLoss(loss_weights=cfg.loss_weights)

        self.visibility = cfg.model['decoder'].get('visibility', False)
        if self.visibility:
            visibility_cfg = {k: v for k, v in cfg.model['visibility_head'].items() if k != 'layer_name'}
            self.visibility_head = VisibilityPerLandmark(**visibility_cfg)
            self.vis_criteria = torch.nn.BCEWithLogitsLoss()

        self.viz_dir = viz_dir
        self.validation_outputs = []
        
        self.iterations = 0
        self.joints_loss_after_n_iters = cfg.model.joints_loss_after_n_iters

    
    def forward(self, x):
        with torch.no_grad() if self.freeze_backbone else torch.enable_grad():
            features = self.backbone(x)
        
        heatmap = self.decoder(features)
        heatmap = torch.sigmoid(heatmap)
        
        joints2d = keypoints_from_heatmap_soft_argmax(heatmap.clamp(min=1e-6).log(), scale_factor=4)
        joints2d[..., 0] /= self.image_size[0]
        joints2d[..., 1] /= self.image_size[1]
        joints2d = joints2d * 2 - 1

        pred = dict(
            heatmap=heatmap,
            joints2d=joints2d
        )

        if self.visibility:
            visibility = self.visibility_head(features, self.backbone.pos_embed)
            pred["visibility"] = visibility

        self.iterations += 1
        return pred
    
    def training_step(self, batch, batch_idx):
        target = batch
        images = target['image'].to(self.device)
        target_weights = target['target_weight'].to(self.device)

        # For heatmap, we don't care joints around the corner or outside of the image
        # TODO: Check why this was not taken by Dataset
        inbound_mask = (target['joints'].abs() < 0.95).all(dim=-1)
        target_weights_for_joint = target_weights.clone()
        target_weights_for_joint[~inbound_mask] = 0.0
        
        # Forward
        pred = self(images)

        # # Check
        # joints2d_recon = keypoints_from_heatmap_soft_argmax(target["target"].clamp(min=1e-6).log())
        # joints2d_recon[..., 0] /= self.image_size[0]
        # joints2d_recon[..., 1] /= self.image_size[1]
        # joints2d_recon = joints2d_recon * 2 - 1
        
        # Compute losses
        loss = dict()
        if self.iterations > self.joints_loss_after_n_iters:
        # if True:
            loss = self.joints_criteria(pred, target, target_weights_for_joint)
        else:
            loss['loss'] = torch.tensor(0.0).to(pred['joints2d'].device)
            loss['loss_joints2d'] = torch.tensor(0.0).to(pred['joints2d'].device)
        loss['loss_heatmap'] = self.hm_criteria(pred["heatmap"], target["target"], target_weights)
        loss['loss'] = loss['loss'] + loss['loss_heatmap']
        
        if self.visibility:
            loss['loss_visibility'] = self.vis_criteria(pred['visibility'], target["joints_visibility"]) * 2e-2
            loss['loss'] = loss['loss'] + loss['loss_visibility']

        steps = 10 if self.cfg.debug_mode else 2000
        train_path = os.path.join(self.viz_dir, 'train')
        os.makedirs(train_path, exist_ok=True)
        if self.global_step > 0 and self.global_step % steps == 0:
        # if self.global_step % 2 == 0:
            with torch.no_grad():
                video_path = compare_results_denseldmks2d(images, pred, target, self.global_step, train_path, self.train_data_cfg["normalize_plus_min_one"])
                self.wandb_video_log(video_path, 'train')
        if self.global_step > 0 and self.global_step % self.cfg.log_steps == 0:
            self.tensorboard_logging(loss, self.global_step, train=True)

        optimizer = self.optimizers()
        lr = optimizer.param_groups[0]["lr"]
        lr_backbone = 0.0 if self.freeze_backbone else optimizer.param_groups[-1]["lr"]
        
        self.log('train/loss', loss["loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=images.size(0))
        self.log('train/loss_joints2d', loss["loss_joints2d"], on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=images.size(0))
        self.log('train/loss_heatmap', loss["loss_heatmap"], on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=images.size(0))
        self.log('train/lr', lr, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=images.size(0))
        self.log('train/lr_backbone', lr_backbone, on_step=True, on_epoch=False, prog_bar=False, logger=True, batch_size=images.size(0))
        if "loss_visibility" in loss:
            self.log('train/loss_visibility', loss["loss_visibility"], on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=images.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        print("Validation step")
        if self.val_data_cfg['type'] == 'BEDLAM_WD':
            target = batch
            images = target['image'].to(self.device)
            target_weights = target['target_weight'].to(self.device)
            
            pred = self(images)

            loss = self.joints_criteria(pred, target, target_weights)
            loss['loss_heatmap'] = self.hm_criteria(pred["heatmap"], target["target"], target_weights)
            loss['loss'] = loss['loss'] + loss['loss_heatmap']
            if self.visibility:
                loss['loss_visibility'] = self.vis_criteria(pred['visibility'], target["joints_visibility"])
                loss['loss'] = loss['loss'] + loss['loss_visibility']

            self.tensorboard_logging(loss, self.global_step, train=False,)
            self.log('val/loss', loss["loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True,sync_dist=True, batch_size=images.size(0))  # NOTE: , sync_dist=True MAYBE?
            self.log('val/loss_joints2d', loss["loss_joints2d"], on_step=True, on_epoch=True, prog_bar=True, logger=False, sync_dist=True, batch_size=images.size(0))
            self.log('val/loss_heatmap', loss["loss_heatmap"], on_step=True, on_epoch=True, prog_bar=True, logger=False, sync_dist=True, batch_size=images.size(0))
            if "loss_visibility" in loss:
                self.log('val/loss_visibility', loss["loss_visibility"], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=images.size(0))
            self.show_results = [images, pred, target]
            return loss

        val_output = {'val_loss': 0.0}  # Modified to store outputs
        self.validation_outputs.append(val_output)  # Store the output

        return val_output

    def on_validation_epoch_end(self):
        val_path = os.path.join(self.viz_dir, 'val')
        os.makedirs(val_path, exist_ok=True)
        with torch.no_grad():
            images, pred, target = self.show_results
            video_path = compare_results_denseldmks2d(images, pred, target, self.global_step, val_path, self.val_data_cfg["normalize_plus_min_one"])
            self.wandb_video_log(video_path, 'val')
        self.validation_outputs.clear()