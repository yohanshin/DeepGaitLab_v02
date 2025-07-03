import torch
import sys, os
sys.path.append("./")

import cv2
import json
import tqdm
import random
import joblib
import numpy as np
import webdataset as wds

from configs.landmarks import surface, anatomy_v0
from .HumanPoseEstimation import HumanPoseEstimationDataset as Dataset
from ..models.detectors.utils.transform import fliplr_joints, get_affine_transform
from .utils.augmentations import extreme_cropping, augment_image, augment_mask, xyxy2cs


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


class BEDLAM_WD(Dataset):
    def __init__(self,
                 label_path= "/is/cluster/fast/scratch/hcuevas/bedlam_lab/BEDLAM_MASKS_WD",
                 landmark_type="surface",
                 normalize_plus_min_one=False,
                 target_type="keypoints",
                 **kwargs):
        super(BEDLAM_WD, self).__init__(**kwargs)

        self.landmark_type = landmark_type
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

        print(self.is_train)
        print("-"*10)
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

        if self.is_train:
            dataset_tar_list_fn = os.path.join(label_path, "tar_train_list.txt")
        else:
            dataset_tar_list_fn = os.path.join(label_path, "tar_eval_list.txt")

        unique_names = set()
        web_dataset_tars = []
        total_data = 0
        with open(dataset_tar_list_fn, 'r') as f:
            # read list and remove \n
            for line in f:
                tar_fn = line.strip()
                web_dataset_tars.append(os.path.join(label_path, tar_fn))
                scene_name = os.path.dirname(tar_fn)
                if scene_name not in unique_names:
                    # open json
                    with open(os.path.join(label_path, scene_name, "metadata.json"), 'r') as json_file:
                        scene_data = json.load(json_file)
                        total_data += scene_data["count_crop"]
                    unique_names.add(scene_name)
                web_dataset_tars = sorted(web_dataset_tars)

        print("Total data: ", total_data)
        self.web_dataset_tars = web_dataset_tars

        self.legth = total_data
        print(self.legth)
        self.total_valid_ldmks_idxs = self.generate_new_idxs_for_less_ldmks(self.num_joints, self.max_num_joints)

    @staticmethod
    def generate_new_idxs_for_less_ldmks(new_n_ldmks, total_n_ldmks):
        '''
        When we generated the dataset, we used 512 landmarks, 256 for one half and 256 for the other half.
        Thus, we need to generate the new indexes for the landmarks in case we want to use less landmarks.
        '''

        assert new_n_ldmks % 2 == 0, "new_n_ldmks should be even"
        assert total_n_ldmks % 2 == 0, "total_n_ldmks should be even"
        assert new_n_ldmks <= total_n_ldmks, "new_n_ldmks should be less than total_n_ldmks"
        assert new_n_ldmks > 0, "new_n_ldmks should be greater than 0"
        assert total_n_ldmks > 0, "total_n_ldmks should be greater than 0"

        valid_joints_idx = np.arange(total_n_ldmks)
        if new_n_ldmks == total_n_ldmks:
            return valid_joints_idx

        half_total_n_ldmks = total_n_ldmks//2
        half_new_n_ldmks = new_n_ldmks//2
        valid_joint_idxs_right = valid_joints_idx[:half_new_n_ldmks]
        valid_joint_idxs_left = valid_joints_idx[half_total_n_ldmks: half_total_n_ldmks + half_new_n_ldmks]
        total_valid_joints_idx = np.concatenate([valid_joint_idxs_right, valid_joint_idxs_left])
        return total_valid_joints_idx


    def load_tars_as_webdataset(self, train: bool,
            resampled=False,
            epoch_size=None,
            cache_dir=None,
            **kwargs) -> Dataset:
        """
        Loads the dataset from a webdataset tar file.
        """

        def split_data(source):
            for item in source:
                bodies = item["data.pyd"]
                for body, bodies_data in bodies.items():
                    # if bodies_data["person_visitiblity_rate"] < 0.15:
                    if bodies_data["person_visitiblity_rate"] < 0.3:
                        continue
                    yield {
                        "__key__": item["__key__"] + f"_{body}",
                        "jpg": item["jpg"],
                        "data.pyd": bodies_data,
                        "mask.jpg": item["mask.jpg"],
                    }

        # Load the dataset
        if epoch_size is not None:
            resampled = True

        urls= self.web_dataset_tars
        dataset = wds.WebDataset(urls,
                                nodesplitter=wds.split_by_node,
                                # workersplitter=wds.split_by_worker,
                                shardshuffle=True,
                                resampled=resampled,
                                cache_dir=cache_dir,)
        if train:
            dataset = dataset.shuffle(100)
        dataset = dataset.decode('rgb8').rename(jpg='jpg;jpeg;png')

        # Process the dataset
        dataset = dataset.compose(split_data)

        # Process the dataset further
        dataset = dataset.map(lambda x: self.process_webdataset_tar_item(x, train,))
        if epoch_size is not None:
            dataset = dataset.with_epoch(epoch_size)

        print("epoch size: ", epoch_size)

        return dataset

    def process_webdataset_tar_item(self, data, train):
        joints_data = {}
        image = data['jpg'].copy() # Load in RGB format
        mask = data['mask.jpg'].copy()
        person_idx = int(data["data.pyd"]['person_idx'])
        img_name = data['__key__']
        
        if "closeup" in img_name:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
        joints = data["data.pyd"]["vertices2d"].copy()
        joints_vis = data["data.pyd"]["vertex_visibility"].copy()
        
        if self.ldmks2d_idx is not None:
            joints = joints[self.ldmks2d_idx]
            joints_vis = joints_vis[self.ldmks2d_idx]
        else:
            joints = np.matmul(self.smplx2landmarks, joints)
            _joints_vis = np.ones_like(joints[..., :1])
            _joints_vis[self.vis_binary] = joints_vis[self.vis_idx[self.vis_binary]]
            joints_vis = _joints_vis.copy()

        is_random_crop = torch.rand(1).item() if self.is_train else 1.0
        if self.is_train and (is_random_crop < self.extreme_cropping_prob):
            try: 
                bbox, rescale = extreme_cropping(joints, self.body_parts_dict, image.shape[1], image.shape[0])
                c, s = xyxy2cs(bbox, self.aspect_ratio, self.pixel_std)
                s = s * rescale
            except:
                c = np.array(data["data.pyd"]["center"]).copy()
                s = data["data.pyd"]["scale"].copy()
                s = s / 1.2  # remove CLIFF_SCALE_FACTOR
                s = np.array([s,s])    
        else:
            c = np.array(data["data.pyd"]["center"]).copy()
            s = data["data.pyd"]["scale"].copy()
            s = s / 1.2  # remove CLIFF_SCALE_FACTOR
            s = np.array([s,s])

        score = 1
        r = 0

        # Apply data augmentation
        f = False
        if self.is_train:
            sf = self.scale_factor
            rf = self.rotation_factor

            if self.scale:
                s = s * np.clip(random.random() * sf + 1, 1 - sf, 1 + sf)  # A random scale factor in [1 - sf, 1 + sf]

            if self.trans_factor > 0 and random.random() < self.trans_prob:
                # multiplying by self.pixel_std removes the 200 scale
                trans_x = np.random.uniform(-self.trans_factor, self.trans_factor) * self.pixel_std * s[0]
                trans_y = np.random.uniform(-self.trans_factor, self.trans_factor) * self.pixel_std * s[1]
                c[0] = c[0] + trans_x
                c[1] = c[1] + trans_y

            if self.rotate_prob and random.random() < self.rotate_prob:
                r = np.clip(random.random() * rf, -rf * 2, rf * 2)  # A random rotation factor in [-2 * rf, 2 * rf]
            else:
                r = 0

            if self.flip_prob and random.random() < self.flip_prob:
                image = image[:, ::-1, :]
                mask = mask[:, ::-1]
                joints, joints_vis = fliplr_joints(joints, joints_vis, image.shape[1], self.flip_pairs)
                c[0] = image.shape[1] - c[0] - 1
                f = True
            else:
                f = False

            if torch.rand(1).item() < self.img_aug_prob:
                # Numbers taken from bedlam/core/datasets/utils.py get_example
                image = augment_image(image)

        # Apply affine transform on joints and image
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

        mask = ((mask[...,0] == (person_idx+1))).astype(np.float32)
        if self.is_train:
            mask = augment_mask(mask)
        mask = mask[..., None]

        joints = np.concatenate([joints, np.ones((joints.shape[0], 1))], axis=1)
        joints = joints @ trans.T

        # Convert image to tensor and normalize
        if self.transform is not None:  # I could remove this check
            image  = convert_cvimg_to_tensor(image)  # convert from HWC to CHW
            image = (image - self.mean[:, None, None]) / self.std[:, None, None]
            mask = convert_cvimg_to_tensor(mask)  # convert from HWC to CHW

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

        # _image = cvimg.copy()
        # for xy in joints[:, :2]:
        #     cv2.circle(_image, (int(xy[0]), int(xy[1])), color=(0, 255, 0), radius=2, thickness=-1)
        # cv2.imwrite("test.png", _image[..., ::-1])
        
        # scale joints to [0, 1]
        joints[:, 0] /= self.image_size[0]
        joints[:, 1] /= self.image_size[1]

        if self.normalize_plus_min_one:
            # scale joints to [-1, 1]
            joints = joints * 2 - 1

        # Update metadata
        joints_data['image_name'] = img_name
        joints_data['joints'] = joints[self.total_valid_ldmks_idxs]
        joints_data['joints_visibility'] = joints_vis[self.total_valid_ldmks_idxs]
        joints_data['center'] = c
        joints_data['scale'] = s
        joints_data['rotation'] = r
        joints_data['flip'] = f if self.is_train else False
        joints_data['score'] = score
        joints_data['image'] = image.astype(np.float32)
        joints_data['mask'] = mask[:1, ].astype(np.float32)
        joints_data['target'] = target.astype(np.float32)
        joints_data['target_weight'] = target_weight.astype(np.float32)[self.total_valid_ldmks_idxs]
        joints_data['person_visitiblity_rate'] = data["data.pyd"]['person_visitiblity_rate']
        return joints_data

class MixedWebDataset(wds.WebDataset):
    def __init__(self, train_data_cfg, is_train: bool = False) -> None:
        super(wds.WebDataset, self).__init__()
        # dataset_list = cfg.DATASETS.TRAIN if train else cfg.DATASETS.VAL
        dataset = BEDLAM_WD(**train_data_cfg)
        Dataset.registry["BEDLAM_WD"]
        datasets = [dataset.load_tars_as_webdataset(train=is_train, epoch_size=dataset.legth)]
        weights = [1.0] * len(datasets)
        self.append(wds.RandomMix(datasets, weights))

def main():
    batch_size = 1
    dataset = MixedWebDataset(train_data_cfg=dict(normalize_plus_min_one=True), 
                              is_train=True).with_epoch(1000//batch_size)

    mean = np.array([0.485, 0.456, 0.406])[:, None, None]
    std = np.array([0.229, 0.224, 0.225])[:, None, None]

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
    )
    import matplotlib.pyplot as plt
    is_plot = True
    count = 0
    for epoch in tqdm.tqdm(range(10)):
        for batch in dataloader:
            
            images = batch['image']
            masks = batch['mask']
            targets = batch['joints_visibility']
            print(images.shape, epoch, count,)
            if is_plot:
                for i, (img, mask, joints, target_weight) in enumerate(zip(images, masks, batch["joints"].numpy(), targets)):
                    
                    fig, ax = plt.subplots(1, 2)
                    img_norm = (img.numpy() * std + mean).transpose(1, 2, 0)
                    img_norm_ori = img_norm.copy()
                    img_norm = img_norm * mask.numpy().transpose(1, 2, 0)
                    ax[0].imshow(img_norm_ori)
                    print(img_norm.max(), img_norm.min(), img_norm_ori.max(), img_norm_ori.min())
                    ax[1].imshow(img_norm)
                    joints = (joints + 1) / 2
                    joints[:, 0] *= img.shape[2]
                    joints[:, 1] *= img.shape[1]

                    # scatter = ax.scatter(joints[:, 0]*img.shape[2], joints[:, 1]*img.shape[1], c=target_weight[:,0], cmap='plasma', s=1,
                    #                      vmin=0., vmax=2)
                    # colorbar = plt.colorbar(scatter, ax=ax)

                    for vidx, color in enumerate(target_weight):
                        if color > 0:
                            ax[1].scatter(joints[vidx, 0], joints[vidx, 1], c='b', s=1)
                        else:
                            ax[1].scatter(joints[vidx, 0], joints[vidx, 1], c='r', s=1)
                    # ax.scatter(joints[:, 0]*img.shape[2], joints[:, 1]*img.shape[1], c='r', s=1)

                    plt.savefig(f'outputs/test/img_{count:04d}_{batch["person_visitiblity_rate"][i]}.png')
                    plt.close()
                    count += 1
                    import pdb; pdb.set_trace()


if __name__ == '__main__':
    main()