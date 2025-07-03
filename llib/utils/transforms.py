import torch
import numpy as np
from einops import rearrange

def update_intrinsics(bboxes, intrinsics, w=192, h=256):
    # Update after crop
    new_intrinsics = intrinsics.copy()

    # Crop the image
    x1, y1, x2, y2 = np.split(bboxes, 4, axis=1)
    x1, y1, x2, y2 = x1[:, 0], y1[:, 0], x2[:, 0], y2[:, 0]
    W, H = x2 - x1, y2 - y1
    
    cx, cy = intrinsics[:, 0, 2], intrinsics[:, 1, 2]
    new_cx = cx - x1
    new_cy = cy - y1
    new_intrinsics[:, 0, 2], new_intrinsics[:, 1, 2] = new_cx, new_cy
    
    # Resize the image
    fx, fy, cx, cy = new_intrinsics[:, 0, 0], new_intrinsics[:, 1, 1], new_intrinsics[:, 0, 2], new_intrinsics[:, 1, 2]
    new_fx = fx * w / W
    new_fy = fy * h / H
    new_cx = (cx * w / W)
    new_cy = (cy * h / H)
    new_intrinsics[:, 0, 0], new_intrinsics[:, 1, 1], new_intrinsics[:, 0, 2], new_intrinsics[:, 1, 2] = new_fx, new_fy, new_cx, new_cy

    return new_intrinsics

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


def perspective_projection_torch(pt3ds: torch.Tensor, K: torch.Tensor, R: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """Perspective camera projection
    
    Args:
        pt3ds: 3D points in camera coordinate system (n_f, n_j, 3)
        K: Camera intrinsic matrix  (n_f, n_c, 3, 3) or (n_f, 3, 3)
        R: Rotation matrix          (n_f, n_c, 3, 3) or (n_f, 3, 3)
        T: Translation vector       (n_f, n_c, 3) or (n_f, 3)
    """

    if K.dim() == 4:
        pt3ds = pt3ds.unsqueeze(1)
    
    pt3ds = torch.matmul(R, pt3ds.mT) + T.unsqueeze(-1)
    pt2ds = torch.matmul(K, pt3ds).mT
    pt2ds = torch.div(pt2ds[..., :2], pt2ds[..., 2:])

    return pt2ds


def perspective_projection_numpy(pt3ds, K, R, T):
    """Perspective camera projection
    
    Args:
        pt3ds: 3D points in camera coordinate system (n_f, n_j, 3)
        K: Camera intrinsic matrix  (n_f, n_c, 3, 3)
        R: Rotation matrix          (n_f, n_c, 3, 3)
        T: Translation vector       (n_f, n_c, 3)
    """

    return perspective_projection_torch(
        torch.from_numpy(pt3ds).float(),
        torch.from_numpy(K).float(),
        torch.from_numpy(R).float(),
        torch.from_numpy(T).float()
    ).numpy()

def perspective_triangulation_numpy(pt2ds, Ks, Rs, Ts):
    """Triangulation from multiple views

    Args:
        pt2ds: 2D points in camera coordinate system (n_f, n_c, n_j, 2)
        Ks: Camera intrinsic matrix  (n_f, n_c, 3, 3)
        Rs: Rotation matrix          (n_f, n_c, 3, 3)
        Ts: Translation vector       (n_f, n_c, 3)
    """
    
    n_f, n_c, n_j = pt2ds.shape[:3]
    pt3ds = np.zeros((n_f, n_j, 3))
    confs = np.zeros((n_f, n_j))

    projs = []
    for i in range(n_c):
        E = np.concatenate([Rs[:, i], Ts[:, i].reshape(-1, 3, 1)], axis=-1)
        proj = Ks[:, i] @ E
        projs.append(proj)
    projs = np.stack(projs, axis=1)
    if projs.shape[0] == 1: projs = projs.repeat(n_f, axis=0)

    for f in range(n_f):
        for j in range(n_j):
            pt2d = pt2ds[f, :, j, :2]
            conf = pt2ds[f, :, j, -1] # 
            average_conf = (conf[conf > 0]).sum() / n_c

            conf = np.clip(conf, 0, 1)
            
            A = np.repeat(projs[f, :, 2:3], 2, 1) * pt2d.reshape(n_c, 2, 1)
            A -= projs[f, :, :2]
            A *= conf.reshape(-1, 1, 1)

            (u, s, vh) = np.linalg.svd(A.reshape(-1, 4), full_matrices=False)
            pt3d_hom = vh[3, :]
            pt3d = pt3d_hom[:-1] / pt3d_hom[-1]
            pt3ds[f, j] = pt3d
            confs[f, j] = average_conf
    
    return pt3ds



def batch_compute_similarity_transform_torch(S1, S2, return_transform=False):
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    '''
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.permute(0,2,1)
        S2 = S2.permute(0,2,1)
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=-1, keepdims=True)
    mu2 = S2.mean(axis=-1, keepdims=True)

    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = torch.sum(X1**2, dim=1).sum(dim=1)

    # 3. The outer product of X1 and X2.
    K = X1.bmm(X2.permute(0,2,1))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, V = torch.svd(K)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
    Z = Z.repeat(U.shape[0],1,1)
    Z[:,-1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0,2,1))))

    # Construct R.
    R = V.bmm(Z.bmm(U.permute(0,2,1)))

    # 5. Recover scale.
    scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1

    # 6. Recover translation.
    t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))

    if return_transform:
        return R, scale, t

    # 7. Error:
    S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t

    if transposed:
        S1_hat = S1_hat.permute(0,2,1)

    return S1_hat


def dlt_triangulate(points_2d: torch.Tensor, projection_matrices: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """
    Differentiable Direct Linear Transformation (DLT) for 3D triangulation.

    This function triangulates 3D points from a set of 2D points observed from
    multiple camera views. It solves the linear system Ax=0 using Singular Value
    Decomposition (SVD) to find the 3D point in homogeneous coordinates.

    The implementation is fully vectorized and differentiable, making it suitable
    for use in deep learning pipelines.

    Args:
        points_2d (torch.Tensor): A tensor of 2D keypoint coordinates.
            Shape: (B, V, J, 2), where:
                B = batch size
                V = number of views
                J = number of joints/keypoints
                2 = (u, v) coordinates
        projection_matrices (torch.Tensor): A tensor of camera projection matrices.
            Shape: (B, V, 3, 4), where the 3x4 matrix projects 3D points in
            homogeneous coordinates to 2D image coordinates.

    Returns:
        torch.Tensor: A tensor of triangulated 3D points in Euclidean space.
            Shape: (B, J, 3)
    """
    B, V, J, _ = points_2d.shape
    
    # Reshape tensors for batch processing
    # We want to process each joint triangulation independently.
    # New shape: (B*J, V, ...)
    if points_2d.shape[-1] == 3:
        log_sigmas = points_2d[..., -1:]
        confidence = torch.exp(-log_sigmas) + 1e-8 # Equivalent to 1.0 / torch.exp(pred_log_sigmas)
        points_2d = points_2d[..., :2]
    else:
        confidence = torch.ones_like(points_2d[..., :1])
    confidence = confidence * valid.unsqueeze(-2)
    confidence = rearrange(confidence, 'b v j d -> (b j) v d')

    points_2d_reshaped = points_2d.permute(0, 2, 1, 3).reshape(B * J, V, 2)
    projection_matrices_reshaped = projection_matrices.unsqueeze(2).expand(-1, -1, J, -1, -1).permute(0, 2, 1, 3, 4).reshape(B * J, V, 3, 4)

    # Extract u and v coordinates
    u = points_2d_reshaped[:, :, 0]
    v = points_2d_reshaped[:, :, 1]

    # Extract rows of the projection matrices
    P0 = projection_matrices_reshaped[:, :, 0, :]
    P1 = projection_matrices_reshaped[:, :, 1, :]
    P2 = projection_matrices_reshaped[:, :, 2, :]

    # Construct the A matrix for the linear system Ax=0
    # For each view, we get two equations
    A_part1 = confidence * (u.unsqueeze(-1) * P2 - P0)
    A_part2 = confidence * (v.unsqueeze(-1) * P2 - P1)

    # Stack the two equations for each view to form a (2V, 4) matrix per joint
    # We interleave the two parts to get [A_part1_view1, A_part2_view1, A_part1_view2, ...]
    A = torch.cat([A_part1.unsqueeze(2), A_part2.unsqueeze(2)], dim=2)
    A = A.view(B * J, 2 * V, 4)

    # Solve the linear system Ax=0 using SVD
    # The solution is the singular vector corresponding to the smallest singular value.
    # In PyTorch's torch.linalg.svd, this is the last column of V (or last row of Vh)
    _, _, Vh = torch.linalg.svd(A)
    points_4d_homogeneous = Vh[:, -1, :]

    # Dehomogenize the 3D points
    # Add a small epsilon to the denominator for numerical stability
    points_3d = points_4d_homogeneous[:, :3] / (points_4d_homogeneous[:, 3].unsqueeze(-1) + 1e-8)

    # Reshape the output to the final desired shape (B, J, 3)
    points_3d_final = points_3d.view(B, J, 3)

    return points_3d_final