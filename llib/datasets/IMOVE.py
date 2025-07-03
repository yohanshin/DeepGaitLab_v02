import torch
import sys, os
sys.path.append("./")
from glob import glob

import cv2
import json
import tqdm
import random
import joblib
import numpy as np
from skimage.util.shape import view_as_windows

from configs.landmarks import surface, anatomy_v0
from .HumanPoseEstimation import HumanPoseEstimationDataset as Dataset
from ..models.detectors.utils.transform import get_affine_transform
from .utils.augmentations import xyxy2cs


def convert_cvimg_to_tensor(cvimg: np.array):
    """
    Convert image from HWC to CHW format.
    Args:
        cvimg (np.array): Image of shape (H, W, 3) as loaded by OpenCV.
    Returns:
        np.array: Output image of shape (3, H, W).
    """
    # from h,w,c(OpenCV) to c,h,w
    img = cvimg.copy()
    img = np.transpose(img, (2, 0, 1))
    # from int to float
    img = img.astype(np.float32)
    return img

def compute_bbox_info(center, scale, focal_length, image_size):
    bbox_info = np.array([
        center[0] - image_size[0]/2, center[1] - image_size[1]/2, scale
    ])
    bbox_info[:] /= focal_length
    return bbox_info

class IMOVEDataset(Dataset):
    def __init__(self, 
                 label_path="",
                 landmark_type="surface",
                 normalize_plus_min_one=False,
                 fps=5,
                 is_multiview=False,
                 is_multiframe=False,
                 max_num_cameras=10,
                 max_num_frames=16,
                 **kwargs):
        super().__init__(**kwargs)
        self.labels = joblib.load(label_path)
        self.is_multiview = is_multiview
        self.is_multiframe = is_multiframe
        self.landmark_type = landmark_type
        self.max_num_cameras = max_num_cameras
        self.max_num_frames = max_num_frames
        if landmark_type == "surface":
            landmark_cfg = surface
        elif landmark_type == "anatomy_v0":
            landmark_cfg = anatomy_v0

        self.smplx2landmarks = joblib.load(landmark_cfg.subsample_pts_fn)
        if isinstance(self.smplx2landmarks, torch.Tensor):
            self.smplx2landmarks = self.smplx2landmarks.numpy()
        elif isinstance(self.smplx2landmarks, dict):
            self.smplx2landmarks = self.smplx2landmarks["reg"]
        
        if landmark_type == "surface": 
            self.ldmks2d_idx = self.smplx2landmarks.argmax(axis=-1)
        else:
            self.ldmks2d_idx = None
            self.vis_idx = self.smplx2landmarks.argmax(axis=-1)
            self.vis_binary = np.array([self.smplx2landmarks[i, self.vis_idx[i]] > 0.99 for i in range(len(self.vis_idx))])
            self.vis_binary[0] = True

        self.n_landmarks = self.smplx2landmarks.shape[0]

        self.mean = 255 * np.array([0.485, 0.456, 0.406])
        self.std = 255 * np.array([0.229, 0.224, 0.225])

        index_count = 0
        self.index_sets = []
        self.multiview_indices = []
        self.seq_names = []
        for index, camera in enumerate(self.labels['cameras']):
            per_camera_indices = []
            seq_name_parts = self.labels['imagepths'][index][0].split('/')
            seq_name = '_'.join((seq_name_parts[-5], seq_name_parts[-4]))
            self.seq_names.append(seq_name)
            for camera_index, valid in enumerate(camera):
                if valid: 
                    self.index_sets.append((index, camera_index))
                    per_camera_indices.append((index_count, index, camera_index))
                    index_count += 1
            self.multiview_indices.append(per_camera_indices)

        if self.is_multiframe:
            self.prepare_video_batch()
        
        if fps == 5:
            if self.is_multiview:
                self.multiview_indices = self.multiview_indices[::6]
            elif self.is_multiframe:    # Not really 6 FPS, but dropping some videos
                self.seq_indices = self.seq_indices[:len(self.seq_indices) // 6]
            else:
                self.index_sets = self.index_sets[::6]
        
        # import pdb; pdb.set_trace()

    def prepare_video_batch(self):
        self.seq_indices = []
        seq_name = np.array(self.seq_names)
        if isinstance(seq_name, torch.Tensor): seq_name = seq_name.numpy()
        seq_names_unique, group = np.unique(
            seq_name, return_index=True)
        perm = np.argsort(group)
        group_perm = group[perm]
        indices = np.split(
            np.arange(0, len(self.seq_names)), group_perm[1:]
        )
        for cam_idx in range(self.max_num_cameras):
            for idx in range(len(seq_names_unique)):
                indexes = indices[idx]
                if indexes.shape[0] < self.max_num_frames: continue
                chunks = view_as_windows(
                    indexes, (self.max_num_frames), step=self.max_num_frames
                )
                start_finish = chunks[:, (0, -1)].tolist()
                self.seq_indices += [chunk + [cam_idx] for chunk in start_finish]

    def __len__(self):
        if self.is_multiframe:
            return len(self.seq_indices)
        elif self.is_multiview:
            return len(self.multiview_indices)
        return len(self.index_sets)

    def __getitem__(self, index):
        if self.is_multiframe:
            return self.process_multiframe_instance(index)
        elif self.is_multiview:
            return self.process_multiview_instance(index)
        return self.process_single_instance(index)

    def process_multiview_instance(self, index):
        index_info_list = self.multiview_indices[index]
        default_joint_data = dict(
            joints=np.zeros((self.n_landmarks, 2)),
            joints_vis=np.zeros((self.n_landmarks, 1)),
            target=np.zeros(1),
            target_weight=np.zeros((self.n_landmarks, 1)),
            image=np.zeros((3, self.image_size[1], self.image_size[0])),
            mask=np.zeros((1, self.image_size[1], self.image_size[0])),
            bbox_info=np.zeros(3),
            valid=np.array([0]),
            scale=np.zeros(1),
            center=np.zeros(2),
        )

        multiview_joints_data = []
        available_camera_list = [camera_index for _, _, camera_index in index_info_list]
        index_list = [index for index, _, _ in index_info_list]
        for camera_index in range(self.max_num_cameras):
            if camera_index in available_camera_list:
                joints_data = self.process_single_instance(index_list[camera_index])
                joints_data['valid'] = np.array([1])
                multiview_joints_data.append(joints_data)
            else:
                multiview_joints_data.append(default_joint_data.copy())
        
        concatenated_joints_data = {k: np.stack([v[k] for v in multiview_joints_data], axis=0) for k in multiview_joints_data[0].keys()}
        return concatenated_joints_data
    
    def process_multiframe_instance(self, index):
        start_index, end_index, camera_idx = self.seq_indices[index]
        camera_list = self.labels['cameras'][start_index:end_index + 1]
        default_joint_data = dict(
            joints=np.zeros((self.n_landmarks, 2)),
            joints_vis=np.zeros((self.n_landmarks, 1)),
            target=np.zeros(1),
            target_weight=np.zeros((self.n_landmarks, 1)),
            image=np.zeros((3, self.image_size[1], self.image_size[0])),
            mask=np.zeros((1, self.image_size[1], self.image_size[0])),
            bbox_info=np.zeros(3),
            valid=np.array([0]),
            scale=np.zeros(1),
            center=np.zeros(2),
        )
        multiframe_joints_data = []
        multiview_indices = self.multiview_indices[start_index:end_index+1]
        for multiview_index in multiview_indices:
            _index = [i for i, _, j in multiview_index if j == camera_idx]
            if len(_index) == 1:
                _index = _index[0]
                joints_data = self.process_single_instance(index=_index)
                joints_data['valid'] = np.array([1])
                multiframe_joints_data.append(joints_data)
            else:
                multiframe_joints_data.append(default_joint_data.copy())

        concatenated_joints_data = {k: np.stack([v[k] for v in multiframe_joints_data], axis=0) for k in multiframe_joints_data[0].keys()}
        return concatenated_joints_data

    def process_single_instance(self, index):
        joints_data = {}

        multiview_index, camera_index = self.index_sets[index]

        imgpath = self.labels['imagepths'][multiview_index][camera_index]
        cvimg = cv2.cvtColor(cv2.imread(imgpath), cv2.COLOR_BGR2RGB)
        maskpath = self.labels['maskpths'][multiview_index][camera_index]
        mask = np.load(maskpath)[..., None]
        if cvimg.shape[0] != self.image_size[1]:
            cvimg = cv2.resize(cvimg, None, fx=0.5, fy=0.5)
            mask = cv2.resize(mask, None, fx=0.5, fy=0.5)[..., None]
        
        if self.transform is not None:  # I could remove this check
            cvimg  = convert_cvimg_to_tensor(cvimg)  # convert from HWC to CHW
            cvimg = (cvimg - self.mean[:, None, None]) / self.std[:, None, None]
            mask = convert_cvimg_to_tensor(mask)  # convert from HWC to CHW
        
        # Get bbox info
        center = self.labels['center'][multiview_index][camera_index]
        scale = self.labels['scale'][multiview_index][camera_index]
        fl = 2135.0 if camera_index in [1, 2] else 1240.0
        bbox_info = compute_bbox_info(center, scale * self.pixel_std, fl, (1920, 1080))

        # Add dummy joints
        joints = np.zeros((self.n_landmarks, 2))
        joints_vis = np.zeros((self.n_landmarks, 1))
        target_weight = np.zeros((self.n_landmarks, 1))
        target = np.zeros(0)
        joints_data['joints'] = joints.astype(np.float32)
        joints_data['joints_visibility'] = joints_vis.astype(np.float32)
        joints_data['target'] = target.astype(np.float32)
        joints_data['target_weight'] = target_weight.astype(np.float32)
        joints_data['image'] = cvimg.astype(np.float32)
        joints_data['mask'] = mask[:1, ].astype(np.float32)
        joints_data['bbox_info'] = bbox_info.astype(np.float32)
        joints_data['scale'] = scale.astype(np.float32)
        joints_data['center'] = center.astype(np.float32)
        return joints_data
    

def main():
    batch_size = 1

    dataset = IMOVEDataset(label_path="datasets/imove_val_dataset.pkl",
                        #    is_multiview=True,
                           is_multiframe=True,
                           )
    mean = np.array([0.485, 0.456, 0.406])[:, None, None]
    std = np.array([0.229, 0.224, 0.225])[:, None, None]

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True,
        pin_memory=False,
    )
    import matplotlib.pyplot as plt

    is_plot = True
    count = 0
    for epoch in tqdm.tqdm(range(10)):
        for batch in dataloader:
            # images = batch['image']
            # masks = batch['mask']

            # for i, (img, mask) in enumerate(zip(images, masks)):
            #     fig, ax = plt.subplots(1, 2)
            #     img_norm = (img.numpy() * std + mean).transpose(1, 2, 0)
            #     img_norm_ori = img_norm.copy()
            #     img_norm = img_norm * mask.numpy().transpose(1, 2, 0)

            #     ax[0].imshow(img_norm_ori)
            #     ax[1].imshow(img_norm)
            #     plt.savefig(f'outputs/imove/img_{count:04d}.png')
            #     plt.close()
            #     count += 1

            # # Multiview
            # images = batch['image'][0]
            # masks = batch['mask'][0]
            # joints_list = batch["joints"][0].numpy()
            # rows = 2
            # cols = 5
            # fig, ax = plt.subplots(rows, cols)
            # for i, (img, mask, joints) in enumerate(zip(images, masks, joints_list)):
                
            #     img_norm = (img.numpy() * std + mean).transpose(1, 2, 0)
            #     img_norm_ori = img_norm.copy()
            #     img_norm = img_norm * mask.numpy().transpose(1, 2, 0)
            #     row, col = i // cols, i % cols
            #     ax[row, col].imshow(img_norm_ori)
                
            #     joints = (joints + 1) / 2
            #     joints[:, 0] *= img.shape[2]
            #     joints[:, 1] *= img.shape[1]
            #     for xy in joints:
            #         if (xy[0] > 0 and xy[1] > 0) and (xy[0] < img.shape[2] and xy[1] < img.shape[1]):
            #             ax[row, col].scatter(xy[0], xy[1], c='g', s=1)
            # plt.savefig(f'outputs/imove/img_{count:04d}.png')
            # plt.close()
            # count += 1

            # Multiframe
            import pdb; pdb.set_trace()
            images = batch['image'][0]
            masks = batch['mask'][0]
            joints_list = batch["joints"][0].numpy()
            rows = 4
            cols = 4
            fig, ax = plt.subplots(rows, cols)
            for i, (img, mask, joints) in enumerate(zip(images, masks, joints_list)):
                
                img_norm = (img.numpy() * std + mean).transpose(1, 2, 0)
                img_norm_ori = img_norm.copy()
                img_norm = img_norm * mask.numpy().transpose(1, 2, 0)
                row, col = i // cols, i % cols
                ax[row, col].imshow(img_norm_ori)
                
                joints = (joints + 1) / 2
                joints[:, 0] *= img.shape[2]
                joints[:, 1] *= img.shape[1]
                for xy in joints:
                    if (xy[0] > 0 and xy[1] > 0) and (xy[0] < img.shape[2] and xy[1] < img.shape[1]):
                        ax[row, col].scatter(xy[0], xy[1], c='g', s=1)
            plt.savefig(f'outputs/imove/img_{count:04d}.png', dpi=500)
            plt.close()
            count += 1


if __name__ == '__main__':
    main()
