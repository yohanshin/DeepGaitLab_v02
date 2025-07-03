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
from pycocotools.coco import COCO
from skimage.util.shape import view_as_windows

from configs.landmarks import surface, anatomy_v0
from .HumanPoseEstimation import HumanPoseEstimationDataset as Dataset
from ..models.detectors.utils.transform import get_affine_transform
from .utils.augmentations import augment_image, xyxy2cs, cs2xyxy, augment_mask
from ..utils.transforms import update_intrinsics, dlt_triangulate

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

class RandomObjectAugmentator():
    def __init__(self,):
        COCO_DIR = "/is/cluster/fast/sshin/data/COCO"
        
        self.coco_2014_image_dir = os.path.join(COCO_DIR, "images/train2014")
        self.coco_2017_image_dir = os.path.join(COCO_DIR, "images/train2017")
        self.coco_2014_annot_pth = os.path.join(COCO_DIR, "annotations/train2014/instances_train2014.json")
        self.coco_2017_annot_pth = os.path.join(COCO_DIR, "annotations/train2017/instances_train2017.json")

        self.coco_2014 = COCO(self.coco_2014_annot_pth)
        self.coco_2017 = COCO(self.coco_2017_annot_pth)

        self.coco_2014_image_ids = self.coco_2014.getImgIds()
        self.coco_2017_image_ids = self.coco_2017.getImgIds()

        self.coco_2014_image_pths = [os.path.join(self.coco_2014_image_dir, self.coco_2014.loadImgs(img_id)[0]['file_name']) for img_id in self.coco_2014_image_ids]
        self.coco_2017_image_pths = [os.path.join(self.coco_2017_image_dir, self.coco_2017.loadImgs(img_id)[0]['file_name']) for img_id in self.coco_2017_image_ids]

        self.ratio_2014 = len(self.coco_2014_image_ids) / (len(self.coco_2017_image_ids) + len(self.coco_2014_image_ids))
    
    def __call__(self, image, alpha):
        if random.random() < self.ratio_2014:
            idx = random.randint(0, len(self.coco_2014_image_pths) - 1)
            img_info = self.coco_2014.loadImgs(self.coco_2014_image_ids[idx])[0]
            ann_ids = self.coco_2014.getAnnIds(imgIds=[img_info["id"]], iscrowd=False)
            anns = self.coco_2014.loadAnns(ann_ids)
            image_pth = self.coco_2014_image_pths[idx]
            coco_type = "2014"
        else:
            idx = random.randint(0, len(self.coco_2017_image_pths) - 1)
            img_info = self.coco_2017.loadImgs(self.coco_2017_image_ids[idx])[0]
            ann_ids = self.coco_2017.getAnnIds(imgIds=[img_info["id"]], iscrowd=False)
            anns = self.coco_2017.loadAnns(ann_ids)
            image_pth = self.coco_2017_image_pths[idx]
            coco_type = "2017"
        
        obj_image = cv2.imread(image_pth)
        obj_image = cv2.cvtColor(obj_image, cv2.COLOR_BGR2RGB)
        
        num_objects = random.randint(1, 3)
        selected_anns = random.sample(anns, min(num_objects, len(anns)))
        
        for ann in selected_anns:
            if coco_type == "2014":
                mask = self.coco_2014.annToMask(ann)
            else:
                mask = self.coco_2017.annToMask(ann)
            
            bbox = np.array(ann["bbox"])
            bbox = bbox.astype(np.int32)
            x, y, w, h = bbox
            
            _obj_image = obj_image[y:y+h, x:x+w]
            obj_mask = mask[y:y+h, x:x+w] * 255

            min_scale = image.shape[0] * 0.05
            max_scale = image.shape[0] * 0.5
            scale = np.random.uniform(min_scale, max_scale)
            scale_factor = scale / max(h, w)
            new_h = int(h * scale_factor)
            new_w = int(w * scale_factor)
            if min(new_w, new_h) <= 0:
                continue
            
            _obj_image = cv2.resize(_obj_image, (new_w, new_h))
            obj_mask = cv2.resize(obj_mask, (new_w, new_h))

            rotation = np.random.uniform(-45, 45)
            M = cv2.getRotationMatrix2D((new_w/2, new_h/2), rotation, 1)
            _obj_image = cv2.warpAffine(_obj_image, M, (new_w, new_h))
            obj_mask = cv2.warpAffine(obj_mask, M, (new_w, new_h))

            x = random.randint(0, image.shape[1] - new_w)
            y = random.randint(0, image.shape[0] - new_h)
            if np.random.random() < 0.5:
                # Put object behind the perosn
                org_mask = alpha[y:y+new_h, x:x+new_w].copy()
                obj_mask = np.logical_and(obj_mask > 100, org_mask < 100).astype(np.uint8) * 255
                
                _alpha = obj_mask[..., None] / 255
                image[y:y+new_h, x:x+new_w] = image[y:y+new_h, x:x+new_w] * (1 - _alpha) + _obj_image * _alpha
            
            else:
                # Put object behind the perosn
                _alpha = obj_mask[..., None] / 255
                image[y:y+new_h, x:x+new_w] = image[y:y+new_h, x:x+new_w] * (1 - _alpha) + _obj_image * _alpha

                org_mask = (alpha[y:y+new_h, x:x+new_w] / 255).astype(bool)
                _alpha = _alpha > 0.5
                org_mask = np.logical_and(org_mask, ~_alpha[..., 0])
                alpha[y:y+new_h, x:x+new_w] = org_mask.astype(np.uint8) * 255

        return image, alpha


class BEDLAMLABDataset(Dataset):
    def __init__(self, 
                 label_path="",
                 landmark_type="surface",
                 normalize_plus_min_one=False,
                 target_type="keypoints",
                 obj_prob=0.5,
                 n_splits_per_epoch=5,
                 is_multiview=False,
                 is_multiframe=False,
                 max_num_cameras=10,
                 max_num_frames=16,
                 **kwargs):
        super(BEDLAMLABDataset, self).__init__(**kwargs)

        self.labels = joblib.load(label_path)
        self.is_multiview = is_multiview
        self.is_multiframe = is_multiframe
        self.obj_prob = obj_prob
        self.landmark_type = landmark_type
        self.n_splits_per_epoch = n_splits_per_epoch
        self.max_num_cameras = max_num_cameras
        self.max_num_frames = max_num_frames
        if landmark_type == "surface":
            landmark_cfg = surface
        elif landmark_type == "anatomy_v0":
            landmark_cfg = anatomy_v0

        self.target_type = target_type
        self.flip_pairs = landmark_cfg.flip_pairs
        joints_weight = [1] * self.max_num_joints
        self.joints_weight = np.array(joints_weight).reshape(self.max_num_joints, 1)

        self.body_parts_dict = landmark_cfg.body_parts_dict
        self.body_idx = landmark_cfg.body_idx
        self.parts2body_idx = np.argsort(self.body_idx)

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
        self.mean = 255 * np.array([0.485, 0.456, 0.406])
        self.std = 255 * np.array([0.229, 0.224, 0.225])
        self.normalize_plus_min_one = normalize_plus_min_one
        self.n_landmarks = self.smplx2landmarks.shape[0]

        index_count = 0
        self.index_sets = []
        self.multiview_indices = []
        self.seq_names = []
        for index, camera in enumerate(self.labels['cameras']):
            per_camera_indices = []
            seq_name_parts = self.labels['imgpaths'][index].split('/')
            self.seq_names.append(seq_name_parts[-3])
            for camera_index, valid in enumerate(camera):
                if valid: 
                    self.index_sets.append((index, camera_index))
                    per_camera_indices.append((index_count, index, camera_index))
                    index_count += 1
            self.multiview_indices.append(per_camera_indices)

        if self.is_multiframe:
            self.prepare_video_batch()

        self.objmask = RandomObjectAugmentator()
        # self.objmask = None
        
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
        for idx in range(len(seq_names_unique)):
            indexes = indices[idx]
            if indexes.shape[0] < self.max_num_frames: continue
            chunks = view_as_windows(
                indexes, (self.max_num_frames), step=4
            )
            start_finish = chunks[:, (0, -1)].tolist()
            self.seq_indices += start_finish
    
    def __len__(self):
        # return len(self.labels['imgpaths'])
        if self.is_multiframe:
            return len(self.seq_indices)
        if self.is_multiview:
            return len(self.multiview_indices[::self.n_splits_per_epoch])
        return len(self.index_sets[::self.n_splits_per_epoch])

    def __getitem__(self, index):
        if self.is_multiframe:
            return self.process_multiframe_instance(index=index)
        if self.is_multiview:
            return self.process_multiview_instance(_index=index)
        return self.process_single_instance(_index=index)

    def project(self, kpts3d, Es, Ks):
        kpts3d_hom = np.concatenate((kpts3d, np.ones_like(kpts3d[..., -1:])), axis=-1)
        kpts3d_cam = Es @ (kpts3d_hom[None]).transpose(0, 2, 1)
        kpts2d_hom = kpts3d_cam[:, :3] / (kpts3d_cam[:, 2:3] + 1e-9)
        kpts2d = Ks @ kpts2d_hom
        return kpts2d[:, :2].transpose(0, 2, 1)

    def rotate_camera(self, image, img_bbox, K, E, camera_index):
        x1, y1, x2, y2 = img_bbox.copy()
        new_K = K.copy()
        rotate = np.eye(4)
        new_K[0, 0], new_K[1, 1] = K[1, 1], K[0, 0]
        if camera_index == 1:
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            img_bbox = np.array([y1, 1920 - x2, y2, 1920 - x1])
            new_K[0, 2], new_K[1, 2] = K[1, 2], 1920 - K[0, 2]
            rotate[:3, :3] = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        elif camera_index == 2:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            img_bbox = np.array([1080 - y2, x1, 1080 - y1, x2])
            new_K[0, 2], new_K[1, 2] = 1080 - K[1, 2], K[0, 2]
            rotate[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        E = rotate @ E
        K = new_K.copy()
        return image, img_bbox, K, E

    def process_multiframe_instance(self, index):
        start_index, end_index = self.seq_indices[index]
        camera_list = self.labels['cameras'][start_index:end_index + 1]
        n_available_frames = np.sum(camera_list, axis=0)
        camera_idx = np.random.choice(np.arange(len(n_available_frames))[n_available_frames > int(self.max_num_cameras / 2)])

        default_joint_data = dict(
            joints=np.zeros((self.n_landmarks, 2), dtype=np.float32),
            joints_visibility=np.zeros((self.n_landmarks, 1), dtype=np.float32),
            target=np.zeros(1, dtype=np.float32),
            target_weight=np.zeros((self.n_landmarks, 1), dtype=np.float32),
            image=np.zeros((3, self.image_size[1], self.image_size[0]), dtype=np.float32),
            mask=np.zeros((1, self.image_size[1], self.image_size[0]), dtype=np.float32),
            bbox_info=np.zeros(3, dtype=np.float32),
            valid=np.array([0]),
            proj_matrix=np.zeros((3, 4), dtype=np.float32),
            inv_trans=np.zeros((2, 3), dtype=np.float32),
        )

        multiframe_joints_data = []
        multiview_indices = self.multiview_indices[start_index:end_index+1]
        for multiview_index in multiview_indices:
            _index = [i for i, _, j in multiview_index if j == camera_idx]
            if len(_index) == 1:
                _index = _index[0]
                if torch.rand(1).item() < 0.5:
                    img_aug_seed = random.randint(0, 1000000)
                else:
                    img_aug_seed = None
                
                joints_data = self.process_single_instance(None, index=_index, img_aug_seed=img_aug_seed)
                joints_data['valid'] = np.array([1])
                multiframe_joints_data.append(joints_data)
            else:
                multiframe_joints_data.append(default_joint_data.copy())
            
        concatenated_joints_data = {k: np.stack([v[k] for v in multiframe_joints_data], axis=0) for k in multiframe_joints_data[0].keys()}
        return concatenated_joints_data
        
    
    def process_multiview_instance(self, _index, index=None):
        if index is None:
            res = np.random.randint(0, self.n_splits_per_epoch)
            index = (res + self.n_splits_per_epoch * _index) % len(self)
        
        index_info_list = self.multiview_indices[index]
        default_joint_data = dict(
            joints=np.zeros((self.n_landmarks, 2), dtype=np.float32),
            joints_visibility=np.zeros((self.n_landmarks, 1), dtype=np.float32),
            target=np.zeros(1, dtype=np.float32),
            target_weight=np.zeros((self.n_landmarks, 1), dtype=np.float32),
            image=np.zeros((3, self.image_size[1], self.image_size[0]), dtype=np.float32),
            mask=np.zeros((1, self.image_size[1], self.image_size[0]), dtype=np.float32),
            bbox_info=np.zeros(3, dtype=np.float32),
            valid=np.array([0]),
            proj_matrix=np.zeros((3, 4), dtype=np.float32),
            inv_trans=np.zeros((2, 3), dtype=np.float32),
        )
        multiview_joints_data = []
        available_camera_list = [camera_index for _, _, camera_index in index_info_list]
        index_list = [index for index, _, _ in index_info_list]
        valid_i = 0
        for camera_index in range(self.max_num_cameras):
            if camera_index in available_camera_list:
                joints_data = self.process_single_instance(None, index=index_list[valid_i])
                joints_data['valid'] = np.array([1])
                multiview_joints_data.append(joints_data)
                valid_i += 1
            else:
                multiview_joints_data.append(default_joint_data.copy())
        
        concatenated_joints_data = {k: np.stack([v[k] for v in multiview_joints_data], axis=0) for k in multiview_joints_data[0].keys()}
        concatenated_joints_data['joints3d'] = self.labels['landmarks3D'][self.index_sets[index_list[0]][0]]
        return concatenated_joints_data

    def process_single_instance(self, _index, index=None, img_aug_seed=None):
        joints_data = {}
        if index is None:
            res = np.random.randint(0, self.n_splits_per_epoch)
            index = (res + self.n_splits_per_epoch * _index) % len(self)

        multiview_index, camera_index = self.index_sets[index]
        imgpath = self.labels['imgpaths'][multiview_index].replace("camera_id", f"camera_{camera_index:02d}")
        image = np.load(imgpath) # RGB
        img_bbox = self.labels['img_bbox'][multiview_index][camera_index]
        K = self.labels['Ks'][multiview_index][camera_index]
        E = self.labels['Es'][multiview_index][camera_index]
        
        if camera_index in [1, 2]:
            image, img_bbox, K, E = self.rotate_camera(image, img_bbox, K, E, camera_index)
        
        mask = image[..., 3]
        image = image[..., :3]
        # Add random objects
        if random.random() < self.obj_prob and self.objmask is not None:
            image, mask = self.objmask(image, mask)
        
        # Get joints
        joints3d = self.labels['landmarks3D'][multiview_index].copy()
        joints_full = self.project(joints3d, E, K)[0]
        joints = joints_full - img_bbox[:2][None]

        org_shape = np.array(image.shape[:2]).astype(float)
        bbox = np.array([0, 0, org_shape[1], org_shape[0]])
        c, s = xyxy2cs(bbox, aspect_ratio=self.aspect_ratio, pixel_std=self.pixel_std)
        s /= 1.2
        r = 0
        
        # Augment
        sf = 0.2
        if self.scale:
            s = s * np.clip(random.random() * sf + 1, 1 - sf, 1 + sf)  # A random scale factor in [1 - sf, 1 + sf]
        if self.trans_factor > 0 and random.random() < self.trans_prob:
            trans_x = np.random.uniform(-self.trans_factor, self.trans_factor) * self.pixel_std * s[0]
            trans_y = np.random.uniform(-self.trans_factor, self.trans_factor) * self.pixel_std * s[1]
            c[0] = c[0] + trans_x
            c[1] = c[1] + trans_y

        if torch.rand(1).item() < self.img_aug_prob:
            # Numbers taken from bedlam/core/datasets/utils.py get_example
            image = augment_image(image, seed=img_aug_seed)

        trans = get_affine_transform(c, s, self.pixel_std, r, self.image_size)
        image = cv2.warpAffine(
            image,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR
        )
        mask = cv2.warpAffine(
            mask,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_NEAREST
        )
        mask = augment_mask(mask.astype(np.float32) / 255.0)
        mask = mask[..., None]
        
        joints = np.concatenate([joints, np.ones((joints.shape[0], 1))], axis=1)
        joints = joints @ trans.T

        # inversion
        inv_trans = get_affine_transform(img_bbox[:2] + c, s, self.pixel_std, r, self.image_size, inv=1)

        if 'joints_visibility' in self.labels:
            joints_vis = self.labels['joints_visibility'][multiview_index][camera_index].copy()
        else:
            joints_vis = np.ones((joints.shape[0], 1))

        valid_joints_x = (np.zeros(joints.shape[0]) <= joints[:, 0]) & (joints[:, 0] < self.image_size[0])
        valid_joints_y = (np.zeros(joints.shape[0]) <= joints[:, 1]) & (joints[:, 1] < self.image_size[1])
        valid_joints = (valid_joints_x & valid_joints_y)*1
        valid_joints = valid_joints[:, None]
        joints_vis = joints_vis * valid_joints # NOTE: DOUBLE CHECK THIS!!!!

        # get joints outside the image
        target_weight = valid_joints.copy().astype(np.float32)
        if joints[valid_joints[:,0]==0].shape[0] > 0:
            # normalize joints outside the image to [-1, 1]
            outside_joints = 2*(joints[valid_joints[:,0]==0]/self.image_size - 0.5)
            # calculate the distance of the joints to the center
            outside_joints = np.linalg.norm(outside_joints, axis=1)
            beta = 2. #0.25 #4
            outside_joints= np.exp(-beta*np.abs(outside_joints-1))
            target_weight[valid_joints[:,0]==0, 0] = outside_joints
        
        if self.target_type == "keypoints":
            target = np.array([0])
        else:
            target, _ = self._generate_target(joints, joints_vis[self.total_valid_ldmks_idxs], score_invis=1.0)

        # _image = image.copy()
        # for xy in joints[:, :2]:
        #     cv2.circle(_image, (int(xy[0]), int(xy[1])), color=(0, 255, 0), radius=2, thickness=-1)
        # cv2.imwrite("test.png", _image[..., ::-1])

        # Convert image to tensor and normalize
        if self.transform is not None:  # I could remove this check
            image  = convert_cvimg_to_tensor(image)  # convert from HWC to CHW
            image = (image - self.mean[:, None, None]) / self.std[:, None, None]
            mask = convert_cvimg_to_tensor(mask)  # convert from HWC to CHW

        # scale joints to [0, 1]
        joints[:, 0] /= self.image_size[0]
        joints[:, 1] /= self.image_size[1]
        joints = joints * 2 - 1

        # Get bbox info
        center = img_bbox[:2] + c
        bbox_info = compute_bbox_info(center, s.max() * self.pixel_std, K[0, 0], (1920, 1080))

        xyxy = cs2xyxy(center, s, self.pixel_std)
        cropped_K = update_intrinsics(xyxy[None], K[None], w=self.image_size[0], h=self.image_size[1])[0]
        proj_matrix = cropped_K @ E[:3]
        
        joints_data['joints'] = joints.astype(np.float32)
        joints_data['joints_visibility'] = joints_vis.astype(np.float32)
        joints_data['image'] = image.astype(np.float32)
        joints_data['mask'] = mask[:1, ].astype(np.float32)
        joints_data['target'] = target.astype(np.float32)
        joints_data['bbox_info'] = bbox_info.astype(np.float32)
        joints_data['target_weight'] = target_weight.astype(np.float32)
        joints_data['proj_matrix'] = proj_matrix.astype(np.float32)
        joints_data['inv_trans'] = inv_trans.astype(np.float32)
        return joints_data

def main():
    batch_size = 4

    dataset = BEDLAMLABDataset(
        label_path="datasets/bedlamlab3d_cmu_processed_anatomy_v0.pth",
        # is_multiview=True,
        is_multiframe=True,
        max_num_frames=64,
        num_joints=58, 
        total_num_joints=58,
        landmark_type="anatomy_v0")
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

    count = 0
    for epoch in tqdm.tqdm(range(10)):
        for batch in dataloader:
            # images = batch['image']
            # masks = batch['mask']
            # for i, (img, mask, joints) in enumerate(zip(images, masks, batch["joints"].numpy())):
            #     fig, ax = plt.subplots(1, 3)
            #     img_norm = (img.numpy() * std + mean).transpose(1, 2, 0)
            #     img_norm_ori = img_norm.copy()
            #     img_norm = img_norm * mask.numpy().transpose(1, 2, 0)
                
            #     joints = (joints + 1) / 2
            #     joints[:, 0] *= img.shape[2]
            #     joints[:, 1] *= img.shape[1]
            #     ax[0].imshow(img_norm_ori)
            #     ax[1].imshow(img_norm)
            #     ax[2].imshow(img_norm_ori)
            #     for xy in joints:
            #         ax[2].scatter(xy[0], xy[1], c='g', s=1)
            #     plt.savefig(f'outputs/bedlamlab/img_{count:04d}.png')
            #     plt.close()
            #     count += 1
            
            # # Check triangulation is correct
            # points_2d = batch['joints']
            # points_2d = (points_2d + 1) / 2
            # points_2d[..., 0] *= batch['image'][0].shape[3]
            # points_2d[..., 1] *= batch['image'][0].shape[2]
            # proj_matrix = batch['proj_matrix']
            # valid = batch['valid'].bool()
            # points_3d = dlt_triangulate(points_2d, proj_matrix, valid)
            # points_3d_ = batch['joints3d']
            
            
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
                
            #     joints = (joints + 1) / 2
            #     joints[:, 0] *= img.shape[2]
            #     joints[:, 1] *= img.shape[1]
            #     # ax[row, col].imshow(img_norm)
            #     ax[row, col].imshow(img_norm_ori)
            #     # for xy in joints:
            #     #     ax[row, col].scatter(xy[0], xy[1], c='g', s=1)
            #     ax[row, col].set_xticks([])
            #     ax[row, col].set_yticks([])
            
            # plt.tight_layout()
            # plt.savefig(f'outputs/bedlamlab/img_{count:04d}.png')
            # plt.close()
            # count += 1

            # Multiframe
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
                
                joints = (joints + 1) / 2
                joints[:, 0] *= img.shape[2]
                joints[:, 1] *= img.shape[1]
                # ax[row, col].imshow(img_norm)
                ax[row, col].imshow(img_norm_ori)
                # for xy in joints:
                #     ax[row, col].scatter(xy[0], xy[1], c='g', s=1)
                ax[row, col].set_xticks([])
                ax[row, col].set_yticks([])
            
            plt.tight_layout()
            plt.savefig(f'outputs/bedlamlab/img_{count:04d}.png')
            plt.close()
            count += 1

if __name__ == '__main__':
    main()