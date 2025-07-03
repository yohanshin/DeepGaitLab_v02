import cv2
import torch
import numpy as np


def normalize_points(pt2d, K):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    # Normalize points
    x_normalized = (pt2d[..., 0] - cx) / fx
    y_normalized = (pt2d[..., 1] - cy) / fy

    return np.stack([x_normalized, y_normalized], axis=-1)
    
def denormalize_points(pt2d, K):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x_pixel = pt2d[..., 0] * fx + cx
    y_pixel = pt2d[..., 1] * fy + cy

    return np.stack([x_pixel, y_pixel], axis=-1)


def undistort_points(pt2d, dist, intrinsics, max_iterations=10) -> torch.Tensor:
    """
    Undistort 2D keypoints using distortion parameters.

    Args:
        pt2d: (num_frames, num_joints, 2) 2D keypoints in image coordinates.
        dist: (5,) Distortion coefficients [k1, k2, p1, p2, k3].
        intrinsics: (3, 3) Camear Intrinsics
        max_iterations: Maximum number of iterations for iterative refinement.

    Returns:
        torch.Tensor: (num_frames, num_joints, 2) Undistorted 2D keypoints.
    """
    
    
    k1, k2, p1, p2, k3 = dist  # Radial and tangential distortion coefficients

    # Start with the original points as initial guess
    pt2d = normalize_points(pt2d.copy(), intrinsics)
    undistorted = pt2d.copy()  # (num_frames, num_joints, 2)

    for _ in range(max_iterations):
        x = undistorted[..., 0]
        y = undistorted[..., 1]
        r2 = x**2 + y**2  # Radial distance squared

        # Compute radial distortion
        radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3

        # Compute tangential distortion
        x_tangential = 2 * p1 * x * y + p2 * (r2 + 2 * x**2)
        y_tangential = p1 * (r2 + 2 * y**2) + 2 * p2 * x * y

        # Apply distortion correction
        x_undistorted = (pt2d[..., 0] - x_tangential) / radial
        y_undistorted = (pt2d[..., 1] - y_tangential) / radial

        # Update undistorted points
        undistorted[..., 0] = x_undistorted
        undistorted[..., 1] = y_undistorted

    return denormalize_points(undistorted, intrinsics)


def undistort_image_and_bbox(image, bbox, camera_matrix, dist_coeffs):
   # Undistort image
   undistorted_img = cv2.undistort(image, camera_matrix, dist_coeffs)
   
   # Get bbox coordinates
   cx, cy = bbox
   
   # Create points for the four corners of the bbox
   bbox_points = np.array([
       [cx, cy],
   ], dtype=np.float32).reshape(-1, 1, 2)
   
   # Undistort the points
   undistorted_points = cv2.undistortPoints(
       bbox_points, 
       camera_matrix, 
       dist_coeffs, 
       P=camera_matrix
   )
   
   # Reshape and get the new bbox coordinates
   undistorted_points = undistorted_points.reshape(-1, 2)
   
   # Get the min and max to form the new bbox
   undist_cx = undistorted_points[0, 0]
   undist_cy = undistorted_points[0, 1]
   
   undistorted_bbox = [undist_cx, undist_cy]
   
   return undistorted_img, undistorted_bbox


def do_undistortion(pts_2d, cameras):
    undistorted_pts_2d = pts_2d.copy()
    for cam_i in range(pts_2d.shape[1]):
        pt_2d = pts_2d[:, cam_i].copy()
        dist = cameras['dists'][cam_i].copy()
        K = cameras['Ks'][cam_i].copy()
        undistorted_pt_2d = undistort_points(pt_2d, dist, K)
        undistorted_pt_2d = np.concatenate((undistorted_pt_2d, pt_2d[..., -1:]), axis=-1)
        undistorted_pts_2d[:, cam_i] = undistorted_pt_2d.copy()
    
    return undistorted_pts_2d
