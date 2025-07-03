import cv2
import torch
import numpy as np
from tqdm import trange


def do_projection(pt_3d, cameras):
    """
    Project a 3D point into 2D using camera parameters.

    Args:
        pt_3d (np.ndarray): Shape (3,), 3D point.
        cameras (dict): Camera parameters dictionary.
        idx (int): Index of the camera to use.

    Returns:
        np.ndarray: Projected 2D point, shape (2,).
    """
    K = cameras['Ks']
    R = cameras['Rs']
    T = cameras['Ts']
    P = K @ np.concatenate((R, T[..., None]), axis=-1)
    
    pt_3d_hom = np.concatenate((pt_3d, np.ones_like(pt_3d[..., :1])), axis=-1)
    reprojected_hom = np.einsum('bij,nj->bni', P, pt_3d_hom)
    pt_2d = reprojected_hom[..., :2] / reprojected_hom[..., -1:]
    return pt_2d


def do_triangulation(pts_2d: np.ndarray, cameras: dict) -> np.ndarray:
    """
    Triangulate a 3D point from multiple 2D points across different cameras.

    Args:
        pts_2d (np.ndarray): Shape (N_views, 2), 2D points from multiple cameras.
        cameras (dict): Camera parameters dictionary containing:
                        - 'Ks': Intrinsic matrices, shape (N_views, 3, 3)
                        - 'Rs': Rotation matrices, shape (N_views, 3, 3)
                        - 'Ts': Translation vectors, shape (N_views, 3)

    Returns:
        np.ndarray: Triangulated 3D point in Cartesian coordinates, shape (3,).
    """
    N_views = pts_2d.shape[0]
    A = []

    for i in range(N_views):
        u_i = pts_2d[i, ..., 0]
        v_i = pts_2d[i, ..., 1]
        conf = pts_2d[i, ..., 2:]
        
        K = cameras['Ks'][i]
        R = cameras['Rs'][i]
        t = cameras['Ts'][i].reshape(3, 1)

        # Compute the projection matrix P_i = K * [R | t]
        P_i = K @ np.hstack((R, t))  # Shape: (3, 4)

        # Formulate the equations:
        A.append(conf * (u_i[..., None] * P_i[2, :] - P_i[0, :]))
        A.append(conf * (v_i[..., None] * P_i[2, :] - P_i[1, :]))

    if len(A[0].shape) == 1:
        A = np.stack(A, axis=0)
    else:
        A = np.stack(A, axis=1)

    # Solve the homogeneous system A * X = 0 using SVD
    U, S, Vt = np.linalg.svd(A)
    X_homogeneous = Vt[..., -1, :4]  # Solution is the last row of V^T
    
    X_cartesian = X_homogeneous[..., :3] / (X_homogeneous[..., 3:] + 1e-6)

    return X_cartesian


def simple_triangulation(pt2d, cameras, apply_conf=False, *args, **kwargs):
    N_f, N_c, N_j, _ = pt2d.shape
    pt3d = pt2d[:, 0].copy()
    
    if apply_conf:
        confs = np.zeros_like(pt2d[:, 0, :, -1])
    
    for f in trange(N_f, desc='Running triangulation'):  # Iterate over frames
        # Run by entire 3D points in a batch
        if apply_conf:
            conf = pt2d[f, ..., -1].mean(0)
            n_valid = (pt2d[f, ..., -1] > 0).sum(0)
            conf[n_valid < 4] = 0.0
            confs[f] = conf.copy()
        
        triang_pt3d = do_triangulation(pt2d[f, :], cameras)
        pt3d[f] = triang_pt3d.copy()

    if apply_conf:
        confs[np.any(np.isinf(pt3d), axis=-1)] = 0.0
        pt3d[np.isinf(pt3d)] = 0.0
        pt3d = np.concatenate((pt3d, confs[..., None]), axis=-1)
            
    return pt3d


def ransac_triangulation(pt2d, cameras, bboxes, num_samples=20, thr=5e-2, n_cameras=6, run_in_batch=True, apply_conf=False):
    """
    Perform RANSAC-based triangulation for human pose estimation.

    Args:
        pt2d (np.ndarray): Shape (N_f, N_c, N_j, 3), containing 2D points from multiple cameras.
        cameras (dict): Camera parameters dictionary.
        bbox_scale (float): Scale of the bounding box.
        thr (float): Threshold for valid distances.

    Returns:
        float: Mean distance of valid 2D points after projection.
        np.ndarray: Distances array of shape (N_f, N_c, N_j).
    """
    N_f, N_c, N_j, _ = pt2d.shape
    dists = np.zeros((N_f, N_c, N_j))  
    
    pt3d = pt2d[:, 0].copy()

    if apply_conf:
        confs = np.zeros_like(pt2d[:, 0, :, -1])

    for f in trange(N_f, desc='Running RANSAC triangulation'):  # Iterate over frames
        if run_in_batch:
            # Run by entire 3D points in a batch
            best_inliers = 0
            best_point = None
            for _ in range(num_samples):
                cam_indices = torch.randperm(N_c)[:n_cameras]
                subset_cameras = {k: cameras[k][cam_indices] for k in ['Ks', 'Rs', 'Ts']}
                
                triang_pt3d = do_triangulation(pt2d[f, cam_indices, :], subset_cameras)
                proj_pt2d = do_projection(triang_pt3d, cameras)
                
                bbox = bboxes[f, :].copy()
                width = bbox[:, 2] - bbox[:, 0]
                height = bbox[:, 3] - bbox[:, 1]
                scale = np.stack((width * 4/3, height)).max(0)
                error = np.linalg.norm(proj_pt2d - pt2d[f, ..., :2], axis=-1)
                inliers = (error/scale[:, None] < thr).sum()
                
                if inliers > best_inliers:
                    best_inliers = inliers
                    best_point = triang_pt3d.copy()

            pt3d[f] = best_point.copy()
            
        else:
            for j in range(N_j):
                best_point = None
                best_error = np.inf

                bbox = bboxes[f, :].copy()
                width = bbox[:, 2] - bbox[:, 0]
                height = bbox[:, 3] - bbox[:, 1]
                scale = np.stack((width * 4/3, height)).max(0)
                
                for _ in range(num_samples):
                    cam_indices = np.random.choice(N_c, 6, replace=False)
                    subset_cameras = {k: cameras[k][cam_indices] for k in ['Ks', 'Rs', 'Ts']}
                    
                    triang_pt3d = do_triangulation(pt2d[f, cam_indices, [j]], subset_cameras)
                    
                    # triang_pt3d = do_triangulation(pt2d[f, cam_indices, :], subset_cameras)
                    proj_pt2d = do_projection(triang_pt3d[None], cameras)
                    
                    error = np.linalg.norm(proj_pt2d - pt2d[f, :, j:j+1, :2], axis=-1)
                    error = error / scale[:, None]
                    
                    if best_error > error.sum():
                        best_error = error.sum()
                        best_point = triang_pt3d.copy()
            
                pt3d[f, j] = best_point.copy()
                
    return pt3d
