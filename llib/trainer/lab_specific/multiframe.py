import os

import torch
from einops import rearrange

from ..base_trainer import BaseTrainer
from ...models.backbone.vit import ViT
from ...models.head.densekp_head import DecoderPerLandmark
from ...models.head.mask_proc import MaskEmbedding
from ...models.modules.multiframe import MultiFrame
from ...models.losses import JointGNLLLoss, JointsVelocityLoss
from ...models.detectors.utils.visualization import compare_results_denseldmks2d

class DenseKPRegressor(BaseTrainer):
    def __init__(self, cfg, viz_dir=None):
        super(DenseKPRegressor, self).__init__(cfg)

        backbone_cfg = {k: v for k, v in cfg.backbone.items() if not k in ['type', 'name']}
        self.backbone_cfg = backbone_cfg
        self.backbone = ViT(**backbone_cfg).to(self.device)
        self.freeze_backbone = cfg.model.freeze_backbone
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()
        
        decoder_cfg = {k: v for k, v in cfg.model['decoder'].items() if k != 'layer_name'}
        self.decoder = DecoderPerLandmark(**decoder_cfg)
        self.integrator = MultiFrame(**cfg.model['integrator'])

        self.visibility = cfg.model['decoder'].get('visibility', False)
        self.criterion = JointGNLLLoss(loss_weights=cfg.loss_weights)
        self.velocity_criterion = JointsVelocityLoss(loss_weight=cfg.loss_weights.velocity, image_size=cfg.image_size)
        self.vis_criteria = torch.nn.BCEWithLogitsLoss()

        self.use_mask = cfg.model.get('use_mask', False)
        if self.use_mask:
            self.mask_embed = MaskEmbedding(self.backbone_cfg['embed_dim'], self.backbone_cfg['patch_size']//4)
        
        self.viz_dir = viz_dir
        self.validation_outputs = []
        self.show_results = dict(images=[], preds=[], targets=[])

    def _forward(self, x, bbox_info, valid, masks=None):
        B, F = x.shape[:2]
        x = rearrange(x, "B F D H W -> (B F) D H W")
        with torch.no_grad() if self.freeze_backbone else torch.enable_grad():
            features = self.backbone(x)
        
        if self.use_mask and masks is not None:
            masks = rearrange(masks, "B F D H W -> (B F) D H W")
            masks_feats = self.mask_embed(masks)
            features = features + masks_feats
        
        features_mv = self.integrator(features, bbox_info, valid)
        pred = self.decoder(features, self.backbone.pos_embed, aux_feature=features_mv)
        pred['joints2d'] = rearrange(pred['joints2d'], "(B F) J D -> B F J D", B=B, F=F)
        if self.visibility:
            pred['visibility'] = rearrange(pred['visibility'], "(B F) J D -> B F J D", B=B, F=F)
        return pred

    def forward(self, x, bbox_info, valid, masks=None):
        B, F = x.shape[:2]
        x = rearrange(x, "B F D H W -> (B F) D H W")
        with torch.no_grad() if self.freeze_backbone else torch.enable_grad():
            features = self.backbone(x)
        
        if self.use_mask and masks is not None:
            masks = rearrange(masks, "B F D H W -> (B F) D H W")
            masks_feats = self.mask_embed(masks)
            features = features + masks_feats

        features_mv = self.integrator(features, bbox_info, valid)
        
        # Start from here!
        pred = self.decoder(features, self.backbone.pos_embed, aux_feature=features_mv)
        pred['joints2d'] = rearrange(pred['joints2d'], "(B F) J D -> B F J D", B=B, F=F)
        if self.visibility:
            pred['visibility'] = rearrange(pred['visibility'], "(B F) J D -> B F J D", B=B, F=F)
        return pred

    def training_step(self, batch, batch_idx):
        target = batch
        images = target['image'].to(self.device)
        masks = target['mask'].to(self.device)
        target_weights = target['target_weight'].to(self.device)
        valid = target['valid'].to(self.device)
        bbox_info = target['bbox_info'].to(self.device)
        
        # Forward
        pred = self(images, bbox_info, valid, masks)

        # Compute losses
        loss = self.criterion(pred, target, target_weights)
        if self.visibility:
            loss['loss_visibility'] = 0.0
            loss['loss_visibility'] = self.vis_criteria(pred['visibility'], target["joints_visibility"])
            loss['loss'] = loss['loss'] + loss['loss_visibility']
        
        loss['vel'] = self.velocity_criterion(pred['joints2d'], target['joints'], target_weights, bbox_info, target['inv_trans'], valid)

        steps = 10 if self.cfg.debug_mode else 2000
        # steps = 10
        train_path = os.path.join(self.viz_dir, 'train')
        os.makedirs(train_path, exist_ok=True)
        if self.global_step > 0 and self.global_step % steps == 0:
            with torch.no_grad():
                video_path = compare_results_denseldmks2d(images[0], 
                                                          {k: v[0] for k, v in pred.items() if v is not None}, 
                                                          {k: target[k][0] for k in ["joints", "joints_visibility"]}, 
                                                          self.global_step, 
                                                          train_path, 
                                                          self.train_data_cfg["normalize_plus_min_one"]
                )
                self.wandb_video_log(video_path, 'train')
        if self.global_step > 0 and self.global_step % self.cfg.log_steps == 0:
            self.tensorboard_logging(loss, self.global_step, train=True)

        optimizer = self.optimizers()
        lr = optimizer.param_groups[0]["lr"]
        lr_integrator = optimizer.param_groups[-1]["lr"] if self.freeze_backbone else optimizer.param_groups[-2]["lr"]
        lr_backbone = 0.0 if self.freeze_backbone else optimizer.param_groups[-1]["lr"]
        
        self.log('train/loss', loss["loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=images.size(0))
        self.log('train/loss_joints2d', loss["loss_joints2d"], on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=images.size(0))
        self.log('train/loss_sigma', loss["loss_sigma"], on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=images.size(0))
        self.log('train/loss_vel', loss["vel"], on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=images.size(0))
        self.log('lr/lr', lr, on_step=True, on_epoch=False, prog_bar=False, logger=True, batch_size=images.size(0))
        self.log('lr/lr_integrator', lr_integrator, on_step=True, on_epoch=False, prog_bar=True, logger=True, batch_size=images.size(0))
        self.log('lr/lr_backbone', lr_backbone, on_step=True, on_epoch=False, prog_bar=True, logger=True, batch_size=images.size(0))
        if "loss_visibility" in loss:
            self.log('train/loss_visibility', loss["loss_visibility"], on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=images.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        target = batch
        images = target['image'].to(self.device)
        masks = target['mask'].to(self.device)
        valid = target['valid'].to(self.device)
        bbox_info = target['bbox_info'].to(self.device)
        target_weights = target['target_weight'].to(self.device)
        
        with torch.no_grad():
            pred = self(images, bbox_info, valid, masks)

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

        self.show_results["images"].append(images[0])
        self.show_results["preds"].append(pred["joints2d"][0])
        self.show_results["targets"].append(target["joints"][0])
        return loss

        
    def on_validation_epoch_end(self):
        val_path = os.path.join(self.viz_dir, 'val')
        os.makedirs(val_path, exist_ok=True)
        with torch.no_grad():
            images = torch.cat(self.show_results["images"])
            pred = {"joints2d": torch.cat(self.show_results["preds"])}
            target = {"joints": torch.cat(self.show_results["targets"])}
            video_path = compare_results_denseldmks2d(images, pred, target, self.global_step, val_path, self.val_data_cfg["normalize_plus_min_one"])
            self.wandb_video_log(video_path, 'val')
        self.validation_outputs.clear()
        self.show_results = dict(images=[], preds=[], targets=[])