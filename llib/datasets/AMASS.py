from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import torch
import joblib
import numpy as np
from smplx import SMPLX
from skimage.util.shape import view_as_windows
from pytorch3d import transforms as tfs
from torch.nn import functional as F

from configs import constants as _C
from configs.landmarks import surface, anatomy_v0
from .HumanPoseEstimation import Dataset


class SMPLAugmentor():
    noise_scale = 1e-2

    def __init__(self, window_size, augment=True):
        self.window_size = window_size
        self.augment = augment
        
    def __call__(self, pose, betas, trans):
        n_frames = pose.shape[0]

        # Global rotation
        rmat = self.get_global_augmentation()
        global_orient = tfs.axis_angle_to_matrix(pose[:, :3])
        pose[:, :3] = tfs.matrix_to_axis_angle(rmat @ global_orient)
        trans = (rmat.squeeze() @ trans.T).T

        # Shape
        shape_noise = self.get_shape_augmentation(n_frames, betas.shape[-1])
        betas = betas + shape_noise

        return pose, betas, trans

    # def speed_augmentation(self, pose, trans):
    #     l = torch.randint(low=int(self.l_default / self.l_factor), high=int(self.l_default * self.l_factor), size=(1, ))

    #     pose = tfs.matrix_to_rotation_6d(pose)
    #     resampled_pose = F.interpolate(
    #         pose[:l].permute(1, 2, 0), self.l_default, mode='linear', align_corners=True
    #     ).permute(2, 0, 1)
    #     resampled_pose = tfs.rotation_6d_to_matrix(resampled_pose)

    #     resampled_transl = F.interpolate(
    #         trans[:l].permute(1, 2, 0), self.l_default, mode='linear', align_corners=True
    #     ).squeeze(0).T
        
    #     pose = resampled_pose
    #     trans = resampled_transl
    #     betas = betas[:self.l_default]
        
    #     return pose, trans, betas
    
    def get_global_augmentation(self, ):
        """Global coordinate augmentation. Random rotation around z-axis"""
        
        angle_z = torch.rand(1) * 2 * np.pi * float(self.augment)
        aa = torch.tensor([0.0, 0.0, angle_z]).float().unsqueeze(0)
        rmat = tfs.axis_angle_to_matrix(aa)

        return rmat

    def get_shape_augmentation(self, n_frames, d_shape):
        """Shape noise modeling."""
        
        shape_noise = torch.normal(
            mean=torch.zeros((1, d_shape)),
            std=torch.ones((1, d_shape)) * 0.1 * float(self.augment)).expand(n_frames, d_shape)

        return shape_noise



class AMASSDataset(Dataset):
    def __init__(self, 
                 label_pth, 
                 landmark_type="anatomy_v0",
                 window_size=51, 
                 prompt_frames=1,
                 *args, **kwargs):
        
        self.labels = joblib.load(label_pth)
        self.landmark_type = landmark_type
        if landmark_type == "surface":
            landmark_cfg = surface
        elif landmark_type == "anatomy_v0":
            landmark_cfg = anatomy_v0
        self.center_idxs = landmark_cfg.center_idxs
        
        self.smplx2landmarks = joblib.load(landmark_cfg.subsample_pts_fn)
        if isinstance(self.smplx2landmarks, torch.Tensor):
            self.smplx2landmarks = self.smplx2landmarks
        elif isinstance(self.smplx2landmarks, dict):
            self.smplx2landmarks = torch.from_numpy(self.smplx2landmarks["reg"])
            
        if landmark_type == "surface": 
            self.ldmks2d_idx = torch.argmax(self.smplx2landmarks, dim=-1)
        else:
            self.ldmks2d_idx = None

        self.window_size = window_size
        self.prompt_frames = prompt_frames
        self.prepare_video_batch()

        self.body_models = {
            gender: SMPLX(os.path.join(_C.PATHS.BODY_MODEL_DIR, 'smplx'), 
                        num_betas=16, 
                        ext='npz', 
                        flat_hand_mean=True, 
                        use_pca=False,
                        gender=gender).eval()
            for gender in ['male', 'female', 'neutral']
        }
        self.augmentor = SMPLAugmentor(window_size + prompt_frames)
        
    def prepare_video_batch(self):
        self.video_indices = []
        vid_name = self.labels['sequence_idx']
        if isinstance(vid_name, torch.Tensor): vid_name = vid_name.numpy()
        video_names_unique, group = np.unique(
            vid_name, return_index=True)
        perm = np.argsort(group)
        group_perm = group[perm]
        indices = np.split(
            np.arange(0, self.labels['sequence_idx'].shape[0]), group_perm[1:]
        )
        for idx in range(len(video_names_unique)):
            indexes = indices[idx]
            if indexes.shape[0] < self.window_size + self.prompt_frames: continue
            chunks = view_as_windows(
                indexes, (self.window_size + self.prompt_frames), step=self.window_size // 4
            )
            start_finish = chunks[:, (0, -1)].tolist()
            self.video_indices += start_finish

    def __len__(self):
        return len(self.video_indices)

    def forward_smplx(self, pose, betas, trans, gender):
        F = pose.shape[0]
        with torch.no_grad():
            output = self.body_models[gender](
                global_orient=pose[:, :3],
                body_pose=pose[:, 3:66],
                betas=betas[:],
                transl=trans,
                left_hand_pose=pose[:, 75:120],
                right_hand_pose=pose[:, 120:165],
                jaw_pose=pose[:, 66:69],
                leye_pose=pose[:, 69:72],
                reye_pose=pose[:, 72:75],
                expression=torch.zeros((F, 10)).float()
            )
        return output
    
    def __getitem__(self, index):
        start_index, end_index = self.video_indices[index]
        pose = torch.from_numpy(self.labels['poses'][start_index:end_index+1].copy()).float()
        trans = torch.from_numpy(self.labels['trans'][start_index:end_index+1].copy()).float()
        betas = torch.from_numpy(self.labels['betas'][start_index:end_index+1].copy()).float()
        gender = self.labels['gender'][start_index].copy()

        # Augment data
        pose, betas, trans = self.augmentor(pose, betas, trans)

        # Get landmark motion
        output = self.forward_smplx(pose, betas, trans, gender)
        verts = output.vertices
        ldmks3d = torch.matmul(self.smplx2landmarks, verts)
        prompt_noise = torch.randn_like(ldmks3d[:self.prompt_frames]) * 0.01
        ldmks3d[:self.prompt_frames] = ldmks3d[:self.prompt_frames] + prompt_noise
        
        # Normalize 3D ldmks to the initial frame
        center = ldmks3d[:self.prompt_frames, self.center_idxs].mean(1, keepdims=True).mean(0, keepdims=True)
        ldmks3d = ldmks3d - center
        
        prompt_ldmks3d = ldmks3d[:self.prompt_frames]
        ldmks3d = ldmks3d[self.prompt_frames:]
        
        return dict(
            ldmks3d=ldmks3d,
            prompt_ldmks3d=prompt_ldmks3d
        )



if __name__ == '__main__':
    dataset = AMASSDataset(label_pth='datasets/parsed_data/amass_smplx.pth')
    dataset[0]