import torch
import numpy as np


def _cs2xyxy(center, scale, pixel_std=200):
    w = scale[0] * pixel_std
    h = scale[1] * pixel_std
    x1 = center[0] - w * 0.5
    y1 = center[1] - h * 0.5
    x2 = x1 + w
    y2 = y1 + h
    return np.array([x1, y1, x2, y2])


def _xyxy2cs(bbox, pixel_std=200, aspect_ratio=192/256, scale_factor=1.0):
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



def filter_keypoints_2d(keypoints_2d, bboxes, threshold=0.1):
    sx = bboxes[..., 2] - bboxes[..., 0]
    sy = bboxes[..., 3] - bboxes[..., 1]
    scale = np.stack((sx, sy)).max(0)[..., None]
    
    masks = np.zeros_like(keypoints_2d[..., 0])
    masks[1:-1] = np.logical_or(
        np.abs(keypoints_2d[1:-1] - keypoints_2d[:-2])[..., :2].max(-1) > scale[1:-1] * threshold, 
        np.abs(keypoints_2d[2:] - keypoints_2d[1:-1])[..., :2].max(-1) > scale[1:-1] * threshold, )
    
    masks = masks.astype(bool)
    keypoints_2d[masks, -2:] = 0.0
    return keypoints_2d



def smooth_keypoints(keypoints: np.ndarray, dim: int = 3, kernel_size: int = 5, kernel_type: str = "uniform") -> np.ndarray:
    """
    Smooth noisy keypoints using a convolutional filter.

    Args:
        keypoints: (N_frames, N_joints, 3) Noisy keypoints.
        kernel_size: Size of the smoothing kernel (must be odd).
        kernel_type: Type of kernel ("uniform" or "gaussian").

    Returns:
        np.ndarray: (N_frames, N_joints, 3) Smoothed keypoints.
    """
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd for convolution.")
    else:
        pad = kernel_size // 2

    # Define the smoothing kernel
    if kernel_type == "uniform":
        kernel = np.ones(kernel_size) / kernel_size
    elif kernel_type == "gaussian":
        sigma = kernel_size / 6.0  # Approximate rule for Gaussian sigma
        x = np.arange(-(kernel_size // 2), kernel_size // 2 + 1)
        kernel = np.exp(-x**2 / (2 * sigma**2))
        kernel /= kernel.sum()  # Normalize
    else:
        raise ValueError("Unsupported kernel type. Use 'uniform' or 'gaussian'.")

    # Initialize the smoothed keypoints
    smoothed_keypoints = np.zeros_like(keypoints)
    values = keypoints[..., :dim].copy()

    if keypoints.shape[-1] == dim + 1:
        conf = keypoints[..., -1].copy()
    else:
        conf = None

    # Apply convolution along the temporal dimension (axis=0)
    for j in range(keypoints.shape[1]):  # For each joint
        for c in range(dim):  # For x, y, z coordinates
            if conf is not None:
                weighted_coord = values[:, j, c] * conf[:, j]
            
                # 2. Convolve weighted coord
                num = np.convolve(weighted_coord, kernel, mode='same')

                # 3. Convolve confidence
                den = np.convolve(conf[:, j], kernel, mode='same') + 1e-6

                smoothed_keypoints[:, j, c] = num / den
            
            else:
                smoothed_keypoints[:, j, c] = np.convolve(
                    keypoints[:, j, c], kernel, mode="same"
                )
        
        if conf is not None:
            smoothed_keypoints[:, j, -1] = conf[:, j]  
    
    smoothed_keypoints[:pad] = keypoints[:pad].copy()
    smoothed_keypoints[-pad:] = keypoints[-pad:].copy()

    return smoothed_keypoints
