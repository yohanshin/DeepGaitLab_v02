import os

import torch
import torch.nn.functional as F
from ..base_trainer import BaseTrainer
from ...models.head.ldmks_prior_head import SpatioTemporalEncoder
from ...models.diffusion import gaussian_diffusion
from ...models.diffusion.utils import create_gaussian_diffusion
from ...models.diffusion.respace import SpacedDiffusionQuestDiff
from ...models.diffusion.wrapper import ModelWrapper
from ...datasets.AMASS import AMASSDataset

class DiffusionTrainer(BaseTrainer):
    def __init__(self, cfg, device):
        super(DiffusionTrainer, self).__init__(cfg)

        encoder_cfg = {k: v for k, v in cfg.model.encoder.items() if k != 'layer_name'}
        network = SpatioTemporalEncoder(**encoder_cfg)

        self.diffusion_train = create_gaussian_diffusion(cfg, 
                                                gd=gaussian_diffusion, 
                                                return_class=SpacedDiffusionQuestDiff, 
                                                device=device)
        self.diffusion_eval = create_gaussian_diffusion(cfg, 
                                                gd=gaussian_diffusion, 
                                                return_class=SpacedDiffusionQuestDiff, 
                                                device=device,
                                                eval=True)
        self.network = ModelWrapper(network, self.diffusion_train, self.diffusion_eval, device=device)

    def setup(self, stage=None):#prepare_data(self):
        self.train_ds = AMASSDataset(**self.train_data_cfg)
        self.val_ds = None
    
    def configure_optimizers(self):
        optimizer = eval(f'torch.optim.{self.optimizer_cfg["type"]}')(
            self.network.parameters(),
            lr=self.optimizer_cfg['lr'],
            betas=self.optimizer_cfg['betas'],
            weight_decay=self.optimizer_cfg['weight_decay']
        )

        if self.lr_cfg['warmup']:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=self.lr_cfg['warmup_ratio'], 
                end_factor=1.0, 
                total_iters=self.lr_cfg['warmup_iters']
            )

        if self.lr_cfg['policy'] == "step":
            main_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, 
                milestones=self.lr_cfg['milestones'], 
                gamma=self.lr_cfg['gamma'], 
            )
        elif self.lr_cfg['policy'] == "cosine":
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=self.lr_cfg['T_max'], 
                eta_min=self.lr_cfg['eta_min']
            )
        sequential = torch.optim.lr_scheduler.SequentialLR(
            optimizer, 
            [warmup_scheduler, main_scheduler], 
            [self.lr_cfg['warmup_iters']]
        )
        return [optimizer], [{
            'scheduler': sequential,
            'interval': 'step',   # <-- iteration-wise stepping
            'frequency': 1,
        }]
    
    def forward(self, ldmks, prompt, ):
        pred = self.network(dict(repr_clean=ldmks, prompt=prompt))
        return pred

    def training_step(self, batch, batch_idx):
        ldmks = batch['ldmks3d']
        prompt = batch['prompt_ldmks3d']
        pred = self(ldmks, prompt)

        loss = F.mse_loss(pred, ldmks)
        optimizer = self.optimizers()
        lr = optimizer.param_groups[0]["lr"]
        
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=ldmks.size(0))
        self.log('lr/lr', lr, on_step=True, on_epoch=False, prog_bar=True, logger=True, batch_size=ldmks.size(0))
        return dict(loss=loss)