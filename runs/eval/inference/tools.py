import cv2
import scipy
import numpy as np

from llib.models.detectors.utils.transform import get_affine_transform


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


def kp2xyxy(kps, img_size, thresh=0.3, scale=1.2, min_vis_joints=6, aspect_ratio=4/3):
    if kps.shape[-1] == 2:
        mask = np.ones_like(kps[..., 0]).astype(bool)
    else:
        mask = kps[..., -1] > thresh
        kps = kps[..., :2]

    num_vis_joints = mask.sum(-1)
    valid_frames = num_vis_joints > min_vis_joints
    if valid_frames.sum() == 0:
        return False, False

    bbox_out = np.zeros((len(valid_frames), 4))
    kps = kps[valid_frames]
    mask = mask[valid_frames]

    min_xy = np.array([k[m].min(0) for k, m in zip(kps, mask)])[:, :2]
    max_xy = np.array([k[m].max(0) for k, m in zip(kps, mask)])[:, :2]
    center = (max_xy + min_xy) / 2.0
    width, height = np.split(max_xy - min_xy, 2, axis=-1)
    width, height = width[:, 0], height[:, 0]
    if aspect_ratio is not None:
        width = np.max(np.stack((width, height/aspect_ratio)), axis=0)
        height = np.max(np.stack((height, width*aspect_ratio)), axis=0)
    x1 = center[:, 0] - width / 2.0 * scale
    y1 = center[:, 1] - height / 2.0 * scale
    x2 = center[:, 0] + width / 2.0 * scale
    y2 = center[:, 1] + height / 2.0 * scale
    H, W = img_size
    x1 = np.clip(x1, 0, W)
    y1 = np.clip(y1, 0, H)
    x2 = np.clip(x2, 0, W)
    y2 = np.clip(y2, 0, H)
    bbox = np.stack((x1, y1, x2, y2), axis=-1)
    bbox_out[valid_frames] = bbox
    return bbox_out, valid_frames


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


def process_cvimg(cvimg, c, s, image_size, pixel_std=200.0):
    mean = 255 * np.array([0.485, 0.456, 0.406])
    std = 255 * np.array([0.229, 0.224, 0.225])
    
    trans = get_affine_transform(c, s, pixel_std, 0, image_size)
    image = cv2.warpAffine(
        cvimg,
        trans,
        (int(image_size[0]), int(image_size[1])),
        flags=cv2.INTER_LINEAR
    )
    image  = convert_cvimg_to_tensor(image)
    image = (image - mean[:, None, None]) / std[:, None, None]
    return image


def convert_kps_to_full_img(kps, c, s, image_size, pixel_std=200.0):
    kps = (kps + 1)/2.0
    kps[:, 0] *= image_size[0]
    kps[:, 1] *= image_size[1]
    
    trans = get_affine_transform(c, s, pixel_std, 0, image_size, inv=1)
    kps_hom = kps.copy()
    kps_hom[:, -1] = 1.0
    kps_cvt = kps_hom @ trans.T
    kps[:, :2] = kps_cvt[:, :2]
    return kps