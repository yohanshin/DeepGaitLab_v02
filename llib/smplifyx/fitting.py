import os
from copy import deepcopy
import ipdb
import torch
import joblib
import numpy as np
from tqdm import tqdm
from smplx import SMPLX

from human_body_prior.tools.model_loader import load_model
from human_body_prior.models.vposer_model import VPoser
from pytorch3d import transforms as tfs

from configs import constants as _C
from llib.smplifyx.loss_fn import SMPLifyLoss
from llib.utils.transforms import batch_compute_similarity_transform_torch



def guess_init(kpts3d, J_regressor, body_model, torso_idxs):
    kpts3d = torch.from_numpy(kpts3d[..., :3]).to(dtype=J_regressor.dtype, device=J_regressor.device)
    
    output_unposed = body_model()
    with torch.no_grad():
        kpts3d_unposed = torch.matmul(J_regressor, output_unposed.vertices)

    orient, scale, transl = batch_compute_similarity_transform_torch(
        kpts3d_unposed[:, torso_idxs].cpu(), 
        kpts3d[:, torso_idxs].cpu(), 
        return_transform=True
    )

    # SMPLX root rotation induces translation offset. Run it again and fix it.
    output_posed = body_model(global_orient=tfs.rotation_matrix_to_angle_axis(orient).cuda(), transl=transl.squeeze(-1).cuda())
    with torch.no_grad():
        kpts3d_posed = torch.matmul(J_regressor, output_posed.vertices)
    orient_check, scale_check, transl_check = batch_compute_similarity_transform_torch(
        kpts3d_posed[:, torso_idxs].cpu(), 
        kpts3d[:, torso_idxs].cpu(), 
        return_transform=True
    )
    transl = (transl + transl_check).squeeze(-1)
    
    return transl, tfs.rotation_matrix_to_angle_axis(orient)

def rel_change(prev_val, curr_val):
    return (prev_val - curr_val) / max([np.abs(prev_val), np.abs(curr_val), 1])