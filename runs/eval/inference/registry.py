import os
import glob
import torch


def load_model(cfg, device):
    if cfg.model.name == "pose_detr":
        from llib.trainer.detectors.regressor import DenseKPRegressor as ModelCls
    elif cfg.model.name == "pose_heatmap":
        from llib.trainer.detectors.heatmap import DenseKPDetector as ModelCls
    elif cfg.model.name == "pose_softargmax":
        from llib.trainer.detectors.heatmap import DenseKPDetector as ModelCls
    elif cfg.model.name == "pose_multiview":
        from llib.trainer.lab_specific.multiview import DenseKPRegressor as ModelCls
    elif cfg.model.name == "ste_wo_cond":
        from llib.trainer.diffusion.ldmks_prior import DiffusionTrainer as ModelCls

    return ModelCls(cfg, device=device).to(device)


def load_checkpoint(model, ckpt_dir):
    checkpoint_pth_list = sorted(glob.glob(os.path.join(ckpt_dir, "*.ckpt")))
    checkpoint_pth = checkpoint_pth_list[-1]    # The most recent one
    state_dict = torch.load(checkpoint_pth, weights_only=False)['state_dict']
    model.load_state_dict(state_dict, strict=False)
    return model