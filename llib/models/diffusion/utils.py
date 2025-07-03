# Copyright (c) Meta Platforms, Inc. and affiliates.
# Part of this code is based on https://github.com/GuyTevet/motion-diffusion-model
from copy import deepcopy

import torch

from configs import constants as _C
from .respace import space_timesteps

def create_gaussian_diffusion(cfg, gd, return_class, device='', dataset=None, eval=False):
    # default params
    predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
    steps = cfg.model.diffusion.num_steps
    scale_beta = 1.  # no scaling
    timestep_respacing = None if not eval else cfg.model.diffusion.timestep_respacing_eval
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(cfg.model.diffusion.noise_schedule, steps, scale_beta)  # [time_steps]
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    return return_class(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not cfg.model.diffusion.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        dataset=dataset,
        device=device,
    )