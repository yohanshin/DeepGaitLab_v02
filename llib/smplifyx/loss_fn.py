import os.path as osp

import torch
from torch import nn
from configs import constants as _C
from llib.utils.transforms import perspective_projection_torch
from smplx import SMPLX

funcl2 = lambda x: torch.sum(x**2)
funcl1 = lambda x: torch.sum(torch.abs(x**2))

class GMoF(nn.Module):
    def __init__(self, rho=1):
        super(GMoF, self).__init__()
        self.rho = rho

    def extra_repr(self):
        return 'rho = {}'.format(self.rho)

    def forward(self, residual):
        squared_res = residual ** 2
        dist = torch.div(squared_res, squared_res + self.rho ** 2)
        return self.rho ** 2 * dist


class SMPLifyLoss(nn.Module):
    def __init__(self, J_regressor, batch_size, device, 
                 **kwargs):
        super().__init__()
        self.smplx = SMPLX(
            model_path=osp.join(_C.PATHS.BODY_MODEL_DIR, 'smplx'), 
            batch_size=batch_size, num_betas=11, 
        ).to(device).eval()
        self.tt = lambda x: torch.from_numpy(x).float().to(device)
        self.robustifier = GMoF()
        
        self.register_buffer('J_regressor', torch.from_numpy(J_regressor).float())
        self.to(device)

    def keypoints_3d_loss(self, pred_kpts3d, kpts3d, **kwargs):
        conf = kpts3d[..., -1:] if kpts3d.size(-1) > 3 else 1
        diff_square = (pred_kpts3d[:, :, :3] - kpts3d[:, :, :3]) *conf
        return funcl2(diff_square)
    
    def keypoints_2d_loss(self, pred_kpts3d, kpts2d, scales, Ks, Rs, Ts, **kwargs):
        def gmof(squared_res, sigma_squared):
            """
            Geman-McClure error function
            """
            return (sigma_squared * squared_res) / (sigma_squared + squared_res)
        
        # Projection
        conf = kpts2d[..., -1:] if kpts2d.size(-1) > 3 else 1
        pred_kpts2d = perspective_projection_torch(pred_kpts3d, Ks, Rs, Ts)
        residual = (pred_kpts2d - kpts2d[..., :-1]) * conf
        scales = scales.reshape(*residual.shape[:2], 1, 1)
        
        residual = residual / ((1e-3 + scales) * 200)
        squared_res = gmof(residual**2, 200)
        return torch.sum(squared_res)
    
    def smooth_kpts_loss(self, pred_kpts3d, **kwargs):
        def smooth(kpts_est, **kwargs):
            "smooth body"
            kpts_interp = kpts_est.clone().detach()
            kpts_interp[1:-1] = (kpts_interp[:-2] + kpts_interp[2:])/2
            loss = funcl2(kpts_est[1:-1] - kpts_interp[1:-1])
            return loss/(kpts_est.shape[0] - 2)
        
        return smooth(pred_kpts3d)
    
    def smooth_pose_loss(self, poses, **kwargs):
        def smooth(_poses, nViews=1):
            nFrames = _poses.shape[0]
            loss = 0
            for nv in range(nViews):
                poses_ = poses[nv*nFrames:(nv+1)*nFrames, ]
                poses_interp = poses_.clone().detach()
                poses_interp[1:-1] = (poses_interp[1:-1] + poses_interp[:-2] + poses_interp[2:])/3
                loss += funcl2(poses_[1:-1] - poses_interp[1:-1])
            return loss/(nFrames-2)/nViews
        poses = poses[:, :66]
        return smooth(poses)
    
    def regularize_pose_loss(self, pose_embedding):
        return funcl2(pose_embedding)
    
    def regularize_shape_loss(self, betas):
        return funcl2(betas)
    
    def forward(self, output, pose_embedding, kpts2d, kpts3d, scales, Ks, Rs, Ts, loss_weight_dict):
        losses = dict()
        
        pred_kpts3d = torch.matmul(self.J_regressor, output.vertices)
        kpts2d = self.tt(kpts2d)
        kpts3d = self.tt(kpts3d)
        scales = self.tt(scales)
        Ks = self.tt(Ks)
        Rs = self.tt(Rs)
        Ts = self.tt(Ts)

        if 'kp3d' in loss_weight_dict:
            loss = loss_weight_dict['kp3d'] * self.keypoints_3d_loss(pred_kpts3d, kpts3d)
            losses['kp3d'] = loss

        if 'kp2d' in loss_weight_dict:
            loss = loss_weight_dict['kp2d'] * self.keypoints_2d_loss(pred_kpts3d, kpts2d, scales, Ks, Rs, Ts)
            losses['kp2d'] = loss
        
        if 'smooth_kpts' in loss_weight_dict:
            loss = loss_weight_dict['smooth_kpts'] * self.smooth_kpts_loss(pred_kpts3d)
            losses['smooth_kpts'] = loss
        
        if 'smooth_pose' in loss_weight_dict:
            loss = loss_weight_dict['smooth_pose'] * self.smooth_pose_loss(output.full_pose)
            losses['smooth_pose'] = loss
        
        if 'reg_pose' in loss_weight_dict:
            loss = loss_weight_dict['reg_pose'] * self.regularize_pose_loss(pose_embedding)
            losses['reg_pose'] = loss

        if 'reg_shape' in loss_weight_dict:
            loss = loss_weight_dict['reg_shape'] * self.regularize_shape_loss(output.betas)
            losses['reg_shape'] = loss

        total_loss = sum(losses.values())
        self.loss_dict = {k: f'{v.item():.1f}' for k, v in losses.items()}
        return total_loss