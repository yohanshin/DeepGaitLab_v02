"""
Parts of the code are taken or adapted from
https://github.com/mkocabas/EpipolarPose/blob/master/lib/utils/img_utils.py
"""
import torch
import numpy as np
import cv2
import albumentations as A
from albumentations import ImageOnlyTransform


def xyxy2cs(bbox, aspect_ratio, pixel_std, scale_factor=0.75):
    x1, y1, x2, y2 = bbox
    center = np.array([x1 + x2, y1 + y2]) * 0.5
    w = x2 - x1
    h = y2 - y1

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array(
        [w * 1.0 / pixel_std, h * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * scale_factor

    return center, scale


def extreme_cropping(joints, body_parts_dict, img_w, img_h):
    valid_joints_x = (np.zeros(joints.shape[0]) <= joints[:, 0]) & (joints[:, 0] < img_w)
    valid_joints_y = (np.zeros(joints.shape[0]) <= joints[:, 1]) & (joints[:, 1] < img_h)
    valid_joints = (valid_joints_x & valid_joints_y)*1
    valid_joints = valid_joints[:, None]

    valid_body_names = []
    valid_weights = []
    for body_part, idx in body_parts_dict.items():
        if np.sum(valid_joints[idx]) > 20:
            valid_body_names.append(body_part)
            if "body" in body_part:
                valid_weights.append(4)
            else:
                valid_weights.append(1)
    valid_weights = torch.tensor(valid_weights)
    valid_weights = valid_weights/torch.sum(valid_weights)

    random_index = torch.multinomial(valid_weights, 1).item()
    random_body_part = valid_body_names[random_index]
    #random.choice(valid_body_names, p=valid_weights)
    ramdom_body_idx = body_parts_dict[random_body_part]
    keypoints = np.concatenate([joints[ramdom_body_idx], valid_joints[ramdom_body_idx]], axis=-1)
    # x1, y1, x2, y2 from keypoints
    bbox = [np.min(keypoints[:, 0]), np.min(keypoints[:, 1]), np.max(keypoints[:, 0]), np.max(keypoints[:, 1])]
    rescale = 1.2 if "body" in random_body_part else 4
    return bbox, rescale


def augment_image(image):
    aug_comp = [A.Downscale(0.5, 0.9, interpolation=0, p=0.1),
                A.ImageCompression(20, 100, p=0.1),
                A.RandomRain(blur_value=4, p=0.1),
                A.MotionBlur(blur_limit=(3, 15),  p=0.2),
                A.Blur(blur_limit=(3, 9), p=0.1),
                A.RandomSnow(brightness_coeff=1.5,
                snow_point_lower=0.2, snow_point_upper=0.4)]
    aug_mod = [A.CLAHE((1, 11), (10, 10), p=0.2), A.ToGray(p=0.2),
            A.RandomBrightnessContrast(p=0.2),
            A.MultiplicativeNoise(multiplier=[0.5, 1.5],
            elementwise=True, per_channel=True, p=0.2),
            A.HueSaturationValue(hue_shift_limit=20,
            sat_shift_limit=30, val_shift_limit=20,
            always_apply=False, p=0.2),
            A.Posterize(p=0.1),
            A.RandomGamma(gamma_limit=(80, 200), p=0.1),
            A.Equalize(mode='cv', p=0.1)]
    albumentation_aug = A.Compose([A.OneOf(aug_comp,
                                p=0.3),
                                A.OneOf(aug_mod,
                                p=0.3)])
    return albumentation_aug(image=image)['image']


class random_mask_noise(ImageOnlyTransform):
    def __init__(self, drop_prob=0.01, add_prob=0.01, p=0.5):
        super().__init__(p=p)
        self.drop_prob = drop_prob
        self.add_prob = add_prob

    def get_params_dependent_on_data(self, params, data):
        return {"drop_prob": self.drop_prob, "add_prob": self.add_prob}


    def apply(self, img, drop_prob, add_prob, **params):
        noise = np.random.rand(*img.shape)
        img = img.copy()
        img[noise < drop_prob] = 0
        img[noise > 1 - add_prob] = 1
        return img


class random_oval_mask(ImageOnlyTransform):
    def __init__(self, p=0.5):
        super().__init__(p=p)

    def get_params_dependent_on_data(self, params, data):
        return {}

    def apply(self, img, **params):
        img_idx = np.argwhere(img == 1)
        if len(img_idx) == 0:
            return img
        contour_idx = np.random.choice(len(img_idx))
        x, y = img_idx[contour_idx]
        h, w = img.shape

        center = (y, x)
        axes = (np.random.randint(0, w//8), np.random.randint(0, h//8))
        angle = np.random.randint(0, 360)
        cv2.ellipse(img, center, axes, angle, 0, 360, (1,), -1)
        return img


def augment_mask(mask):
    aug_comp = A.Compose([
                random_oval_mask(p=.5),
                A.Morphological(scale=mask.shape[0]//25, operation='dilation', p=0.5),
                A.Morphological(scale=mask.shape[0]//25, operation='erosion', p=0.5),
                random_mask_noise(drop_prob=0.01, add_prob=0.01, p=.1),
                ])

    return aug_comp(image=mask)['image']