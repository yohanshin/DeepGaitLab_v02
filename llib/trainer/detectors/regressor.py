import os

import torch

from ..base_trainer import BaseTrainer
from ...models.backbone.vit import ViT
from ...models.head.densekp_head import DecoderPerLandmark
from ...models.losses import JointGNLLLoss
from ...models.detectors.utils.visualization import compare_results_denseldmks2d

class DenseKPRegressor(BaseTrainer):
    def __init__(self, cfg, viz_dir=None):
        super(DenseKPRegressor, self).__init__(cfg)

        backbone_cfg = {k: v for k, v in cfg.backbone.items() if not k in ['type', 'name']}
        self.backbone_cfg = backbone_cfg
        self.backbone = ViT(**backbone_cfg).to(self.device)
        
        decoder_cfg = {k: v for k, v in cfg.model['decoder'].items() if k != 'layer_name'}
        self.decoder = DecoderPerLandmark(**decoder_cfg)

        self.visibility = cfg.model['decoder'].get('visibility', False)
        self.criterion = JointGNLLLoss(loss_weights=cfg.loss_weights)
        self.vis_criteria = torch.nn.BCEWithLogitsLoss()

        self.viz_dir = viz_dir
        self.validation_outputs = []

    def forward(self, x):
        with torch.no_grad() if self.freeze_backbone else torch.enable_grad():
            features = self.backbone(x)
        
        pred = self.decoder(features, self.backbone.pos_embed)
        return pred

    def training_step(self, batch, batch_idx):
        target = batch
        images = target['image'].to(self.device)
        target_weights = target['target_weight'].to(self.device)
        
        # Forward
        pred = self(images)

        # Compute losses
        loss = self.criterion(pred, target, target_weights)
        if self.visibility:
            loss['loss_visibility'] = 0.0
            loss['loss_visibility'] = self.vis_criteria(pred['visibility'], target["joints_visibility"])
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
        self.log('train/loss_sigma', loss["loss_sigma"], on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=images.size(0))
        self.log('train/lr', lr, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=images.size(0))
        self.log('train/lr_backbone', lr_backbone, on_step=True, on_epoch=False, prog_bar=True, logger=True, batch_size=images.size(0))
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

            loss = self.criterion(pred, target, target_weights)
            if self.visibility:
                loss['loss_visibility'] = 0.0
                loss['loss_visibility'] = self.vis_criteria(pred['visibility'], target["joints_visibility"])
                loss['loss'] = loss['loss'] + loss['loss_visibility']

            self.tensorboard_logging(loss, self.global_step, train=False,)
            self.log('val/loss', loss["loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True,sync_dist=True, batch_size=images.size(0))  # NOTE: , sync_dist=True MAYBE?
            self.log('val/loss_joints2d', loss["loss_joints2d"], on_step=True, on_epoch=True, prog_bar=True, logger=False, sync_dist=True, batch_size=images.size(0))
            self.log('val/loss_sigma', loss["loss_sigma"], on_step=True, on_epoch=True, prog_bar=True, logger=False, sync_dist=True, batch_size=images.size(0))
            if "loss_visibility" in loss:
                self.log('val/loss_visibility', loss["loss_visibility"], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=images.size(0))
            self.show_results = [images, pred, target]
            
            val_path = os.path.join(self.viz_dir, 'val')
            os.makedirs(val_path, exist_ok=True)
            video_path = compare_results_denseldmks2d(images, pred, target, self.global_step, val_path, self.val_data_cfg["normalize_plus_min_one"])
            self.wandb_video_log(video_path, 'val')
            return loss

        val_output = {'val_loss': 0.0}  # Modified to store outputs
        self.validation_outputs.append(val_output)  # Store the output

        return val_output


    # def on_validation_epoch_end(self):
    #     val_path = os.path.join(self.viz_dir, 'val')
    #     os.makedirs(val_path, exist_ok=True)
    #     with torch.no_grad():
    #         images, pred, target = self.show_results
    #         video_path = compare_results_denseldmks2d(images, pred, target, self.global_step, val_path, self.val_data_cfg["normalize_plus_min_one"])
    #         self.wandb_video_log(video_path, 'val')
    #     self.validation_outputs.clear()