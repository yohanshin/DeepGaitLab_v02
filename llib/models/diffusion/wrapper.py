from copy import deepcopy
import torch
import numpy as np
from .resample import create_named_schedule_sampler


class ModelWrapper(torch.nn.Module):
    def __init__(self, network, diffusion_train=None, diffusion_eval=None, device=None, **kwargs):
        super().__init__()
        
        self.network = network
        self.diffusion_train = diffusion_train
        self.diffusion_eval = diffusion_eval

        self.configure_schedule_sampler()
        self.device = device


    def forward(self, batch):
        if self.training:
            return self.train_step(batch)
        else:
            return self.inference(batch)

    def configure_schedule_sampler(self, ):
        if self.diffusion_train is None: return
        
        self.schedule_sampler_type = 'uniform'
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, self.diffusion_train)

    def train_step(self, batch):
        t, weights = self.schedule_sampler.sample(batch['repr_clean'].shape[0], self.device)
        pred = self.diffusion_train.training_step(model=self.network, batch=batch, t=t, noise=None)
        return pred

    def inference(self, batch, grad_type='proj', cond_fn_with_grad=False):
        # TODO: Implement multiple predictions

        model_output = self.diffusion_eval.eval_step(model=self.network, 
                                                     batch=batch, grad_type=grad_type, 
                                                     cond_fn_with_grad=cond_fn_with_grad)
        return model_output