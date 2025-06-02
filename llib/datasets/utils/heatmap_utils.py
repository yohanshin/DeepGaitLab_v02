from typing import List, Optional

import torch
import numpy as np
import torch.nn.functional as F

from llib.models.detectors.utils import transform_preds, keypoints_from_heatmaps

image_size = (384, 512)
heatmap_size = (96, 128)
heatmap_sigma = 2
unbiased_encoding = True

def heatmap_from_keypoints(joints):
    """
    :param joints:  [num_joints, 2]
    :param joints_vis: [num_joints, 1]
    :return: target, target_weight(1: visible, score_invis: invisible)
    """

    num_joints = joints.shape[0]
    
    nooi = 0
    target = np.zeros((num_joints,
                       heatmap_size[1],
                       heatmap_size[0]),
                        dtype=np.float32)
    target_weight = np.ones((num_joints, 1), dtype=np.float32)
    tmp_size = heatmap_sigma * 3

    for joint_id in range(num_joints):
        feat_stride = np.asarray(image_size) / np.asarray(heatmap_size)
        
        if unbiased_encoding:
            mu_x = joints[joint_id][0] / feat_stride[0]
            mu_y = joints[joint_id][1] / feat_stride[1]
            
            ul = [mu_x - tmp_size, mu_y - tmp_size]
            br = [mu_x + tmp_size + 1, mu_y + tmp_size + 1]

            if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] \
                    or br[0] < 0 or br[1] < 0:
                # If not, just return the image as is
                target_weight[joint_id] = 0
                nooi += 1
                continue
            
            size = 2 * tmp_size + 1
            x = np.arange(0, heatmap_size[0], 1, np.float32)
            y = np.arange(0, heatmap_size[1], 1, np.float32)
            y = y[:, np.newaxis]

            if target_weight[joint_id] > 0.5:
                target[joint_id] = np.exp(-((x - mu_x)**2 +
                                        (y - mu_y)**2) /
                                        (2 * heatmap_sigma**2))
                
        
        else:
            mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
            mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] \
                    or br[0] < 0 or br[1] < 0:
                # If not, just return the image as is
                target_weight[joint_id] = 0
                nooi += 1
                continue

            # # Generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            # The gaussian is not normalized, we want the center value to equal 1
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * heatmap_sigma ** 2))

            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
            img_y = max(0, ul[1]), min(br[1], heatmap_size[1])

            v = target_weight[joint_id]
            if v >= 0.5:
                target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    return target


def _gaussian_blur_torch(heatmaps: torch.Tensor, kernel=11):
    """
    Simple Gaussian blur for each channel in [N, K, H, W].
    For brevity, no custom sigma. Adjust as you like.
    """
    # We'll do depthwise conv with groups=K*N or so, but let's keep it simple:
    # A quick way is to reshape: (N*K, 1, H, W) -> conv -> reshape back.
    # We skip dynamic sigma calculation. For a real "unbiased" approach,
    sigma = 0.3 * ((kernel - 1) * 0.5 - 1) + 0.8
    assert kernel % 2 == 1, "kernel must be odd."

    # Create a 2D Gaussian kernel for depthwise convolution
    gauss_1d = torch.arange(kernel, dtype=torch.float32, device=heatmaps.device)
    center = (kernel - 1) / 2
    gauss_1d = torch.exp(-((gauss_1d - center) ** 2) / (2 * sigma ** 2))
    gauss_1d /= gauss_1d.sum()

    gauss_2d = gauss_1d.unsqueeze(1) @ gauss_1d.unsqueeze(0)  # [k, k]
    gauss_2d = gauss_2d.view(1, 1, kernel, kernel)            # [1, 1, k, k]

    N, K, H, W = heatmaps.shape

    # Reshape for depthwise conv
    x = heatmaps.view(N*K, 1, H, W)
    # groups = N*K so each channel is convolved independently
    x_blurred = F.conv2d(
        x, gauss_2d,
        stride=1,
        padding=kernel // 2,
        # groups=N*K
    )
    return x_blurred.view(N, K, H, W)

def _get_max_preds_torch(heatmaps: torch.Tensor):
    """
    Vectorized argmax over each [N, K, H, W] channel.
    Returns:
      preds: (N, K, 2)
      maxvals: (N, K, 1)
    """
    N, K, H, W = heatmaps.shape
    # Flatten to (N, K, H*W)
    hm_reshaped = heatmaps.view(N, K, -1)
    maxvals, idx = torch.max(hm_reshaped, dim=2)  # both (N, K)
    maxvals = maxvals.unsqueeze(-1)  # (N, K, 1)

    # Convert linear idx to (x, y)
    idx_x = idx % W
    idx_y = idx // W
    preds = torch.stack((idx_x, idx_y), dim=2).float()  # (N, K, 2)

    # If maxvals=0, set coords=-1
    mask = (maxvals > 0)
    mask_2d = mask.expand(-1, -1, 2)
    preds = torch.where(mask_2d, preds, torch.tensor(-1.0, device=preds.device))

    return preds, maxvals

def keypoints_from_heatmaps_torch(
    heatmaps: torch.Tensor,
    kernel=5,
    scale_factor: float = 4.0,
    use_taylor: bool = True,
    center=None,
    scale=None,
):
    """
    Vectorized version of "unbiased" decoding (Gaussian blur + log + Taylor),
    returning keypoints in a scaled-up space (e.g., 4x).

    Args:
      heatmaps: (N, K, H, W) float Tensor
      kernel:   Gaussian kernel size (odd)
      scale_factor: how much to scale up from heatmap coords
      use_taylor: whether to do local offset via Hessian-based Taylor expansion

    Returns:
      preds:   (N, K, 2) predicted keypoints in resized image coords
      maxvals: (N, K, 1) confidence
    """
    heatmaps = heatmaps.clone()
    N, K, H, W = heatmaps.shape

    preds, maxvals = _get_max_preds_torch(heatmaps)  # use log-blurred for unbiased
    
    # 1) Gaussian blur
    hm_blurred = _gaussian_blur_torch(heatmaps, kernel=kernel)

    # 2) Log transform
    hm_log = torch.log(torch.clamp(hm_blurred, min=1e-10))

    if use_taylor:
        # Vectorized Taylor refinement
        # (x, y) = preds[...,0], preds[...,1]. We'll do integer indexing.
        px = preds[..., 0].round().long()  # shape [N, K]
        py = preds[..., 1].round().long()

        # Boundary mask: need px in [1, W-2], py in [1, H-2], and maxvals>0
        valid = (
            (px > 1) & (px < (W-2)) &
            (py > 1) & (py < (H-2)) &
            (maxvals[..., 0] > 0)
        )  # shape [N, K]

        valid_idx = valid.nonzero(as_tuple=True)  
        if valid_idx[0].numel() > 0:
            # Subset indexing
            b_v = valid_idx[0]  # shape [V]
            k_v = valid_idx[1]  # shape [V]

            px_v = px[b_v, k_v]  # shape [V]
            py_v = py[b_v, k_v]

            # Gather partial derivatives from hm_log
            dx = 0.5 * (
                hm_log[b_v, k_v, py_v, px_v + 1] - hm_log[b_v, k_v, py_v, px_v - 1]
            )
            dy = 0.5 * (
                hm_log[b_v, k_v, py_v + 1, px_v] - hm_log[b_v, k_v, py_v - 1, px_v]
            )

            dxx = 0.25 * (
                hm_log[b_v, k_v, py_v, px_v + 2]
                - 2.0 * hm_log[b_v, k_v, py_v, px_v]
                + hm_log[b_v, k_v, py_v, px_v - 2]
            )
            dxy = 0.25 * (
                hm_log[b_v, k_v, py_v + 1, px_v + 1]
                - hm_log[b_v, k_v, py_v - 1, px_v + 1]
                - hm_log[b_v, k_v, py_v + 1, px_v - 1]
                + hm_log[b_v, k_v, py_v - 1, px_v - 1]
            )
            dyy = 0.25 * (
                hm_log[b_v, k_v, py_v + 2, px_v]
                - 2.0 * hm_log[b_v, k_v, py_v, px_v]
                + hm_log[b_v, k_v, py_v - 2, px_v]
            )

            # Build derivative & Hessian for each valid point
            dxdy = torch.stack([dx, dy], dim=1).unsqueeze(-1)   # [V, 2, 1]
            dxx_dxy_dyy = torch.stack([dxx, dxy, dxy, dyy], dim=1)  # [V, 4]
            # Reshape => [V, 2, 2]
            hessian = dxx_dxy_dyy.view(-1, 2, 2)
            det = hessian[:, 0, 0] * hessian[:, 1, 1] - hessian[:, 0, 1] * hessian[:, 1, 0]

            # We only invert Hessians with a nonzero det
            invertible_mask = (det != 0)
            invertible_idx = invertible_mask.nonzero(as_tuple=True)[0]
            offset = torch.zeros((px_v.size(0), 2), device=px_v.device)

            if invertible_idx.numel() > 0:
                hessian_good = hessian[invertible_idx]      # [G, 2, 2]
                dxdy_good = dxdy[invertible_idx]            # [G, 2, 1]
                hessian_inv_good = torch.linalg.inv(hessian_good)  # [G, 2, 2]
                offset_good = -torch.matmul(hessian_inv_good, dxdy_good)  # [G, 2, 1]
                offset_good = offset_good.squeeze(-1)  # [G, 2]

                # Store into offset
                offset[invertible_idx] = offset_good

            # Now place the offset back into preds
            # offset shape [V, 2], valid_idx shape [2, V]
            preds[b_v, k_v] += offset        

    # 5) Scale to the resized input space
    if center is not None and scale is not None:
        preds = preds.detach().cpu().numpy()
        maxvals = maxvals.detach().cpu().numpy()
        # Return back to the full image frame
        for i in range(N):
            preds[i] = transform_preds(
                preds[i], center[i], scale[i], [W, H], use_udp=False)
    
    else:
        preds = preds * scale_factor

    return preds, maxvals


def softmax(target: torch.Tensor, dim: List[int] = (-1,)):
    dim = [d if d >= 0 else d + target.ndim for d in dim]
    dim_min, dim_max = min(dim), max(dim)
    flat = target.flatten(start_dim=dim_min, end_dim=dim_max)
    flat_sm = torch.softmax(flat, dim=dim_min)
    return flat_sm.reshape(target.shape)

def linspace(
    start: float,
    stop: float,
    num: int,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    endpoint: bool = True,
):
    start = torch.as_tensor(start, device=device, dtype=dtype)
    stop = torch.as_tensor(stop, device=device, dtype=dtype)

    if endpoint:
        if num == 1:
            return torch.mean(torch.stack([start, stop], dim=0), dim=0, keepdim=True)
        else:
            return torch.linspace(start, stop, num, device=device, dtype=dtype)
    else:
        if num > 1:
            step = (stop - start) / num
            return torch.linspace(start, stop - step, num, device=device, dtype=dtype)
        else:
            return torch.linspace(start, stop, num, device=device, dtype=dtype)

def decode_heatmap(
    inp: torch.Tensor,
    dim: List[int] = (-1,),
    output_coord_dim: int = -1
):
    result = []
    dim = [d if d >= 0 else d + inp.ndim for d in dim]
    for d in dim:
        other = [x for x in dim if x != d]
        summed = torch.sum(inp, dim=other, keepdim=True)  # keep only axis=d
        coords = linspace(0.0, 1.0, inp.shape[d], dtype=inp.dtype, device=inp.device)
        # tensordot over axis d of `summed` and axis 0 of `coords`
        decoded = torch.tensordot(summed, coords, dims=([d], [0]))
        decoded = torch.unsqueeze(decoded, d)
        # now squeeze out all the summed dims
        for axis in sorted(dim, reverse=True):
            decoded = decoded.squeeze(axis)
        result.append(decoded)
    return torch.stack(result, dim=output_coord_dim)

def normalize(H, eps=1e-6):
    # P = H / (H.sum(dim=[..., 0,1]) + eps)
    P = H / (H.sum(dim=-1).sum(dim=-1).unsqueeze(-1).unsqueeze(-1) + eps)
    return P
    

def soft_argmax(inp: torch.Tensor, dim: List[int] = (-1,)):
    return decode_heatmap(softmax(inp, dim=dim), dim=dim)
    # return decode_heatmap(normalize(inp), dim=dim)


def keypoints_from_heatmap_soft_argmax(
    heatmaps: torch.Tensor, 
    scale_factor: int = 4
) -> torch.Tensor:
    """
    :param heatmaps: Tensor of shape (J, H, W) or (N, J, H, W) representing per-joint logits.
    :returns: coords Tensor of shape (J, 2) or (N, J, 2) in **pixel** coordinates (x, y).
    """
    has_batch = (heatmaps.ndim == 4)
    if not has_batch:
        # make it (1, J, H, W)
        heatmaps = heatmaps.unsqueeze(0)

    N, J, H, W = heatmaps.shape
    # run soft-argmax in heatmap-space → normalized [0..1] coords
    coords_norm = soft_argmax(heatmaps, dim=[2, 3])  # (N, J, 2), order = [y_norm, x_norm]

    # map back to pixel indices
    ys = coords_norm[..., 0] * (H - 1)
    xs = coords_norm[..., 1] * (W - 1)
    coords_px = torch.stack([xs, ys], dim=-1)      # (N, J, 2) as (x, y)

    if not has_batch:
        coords_px = coords_px.squeeze(0)            # (J, 2)
    
    coords_px = coords_px * scale_factor
    return coords_px


if __name__ == "__main__":
    from llib.datasets.BEDLAM_WD import MixedWebDataset
    batch_size = 16
    dataset = MixedWebDataset(train_data_cfg=dict(normalize_plus_min_one=True, target_type="heatmap"), 
                              is_train=False).with_epoch(1000//batch_size)

    mean = np.array([0.485, 0.456, 0.406])[:, None, None]
    std = np.array([0.229, 0.224, 0.225])[:, None, None]

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
    )
    import matplotlib.pyplot as plt
    for batch in dataloader:    
        images = batch['image']
        masks = batch['mask']
        
        jointss = batch['joints']
        heatmaps = batch['target']

        for i, (img, joints, heatmap) in enumerate(zip(images, jointss, heatmaps)):
            joints = (joints + 1) / 2
            joints[:, 0] *= img.shape[2]
            joints[:, 1] *= img.shape[1]
            
            _heatmap = heatmap_from_keypoints(joints.numpy())
            
            # Check 1. See if heatmaps --> keypoints argmax works
            joints_recon, prob = keypoints_from_heatmaps_torch(heatmaps=torch.from_numpy(_heatmap).unsqueeze(0), kernel=5)
            joints_recon_argmax = joints_recon.squeeze(0)
            error_argmax = (joints_recon_argmax - joints).abs().norm(dim=-1).mean()
            print(f"Heatmap <-> Keypoints through argmax error: {error_argmax:.2f} mm")
            
            # Check 2. See if heatmaps --> keypoints soft argmax works
            joints_recon_soft_argmax = keypoints_from_heatmap_soft_argmax(torch.from_numpy(_heatmap).float())
            error_soft_argmax = np.linalg.norm(np.abs(joints_recon_soft_argmax - joints.numpy()), axis=-1).mean()
            print(f"Heatmap <-> Keypoints through soft argmax error: {error_soft_argmax:.2f} mm")