from .util import *
from .post_processing import *
from .top_down_eval import *

import torch
import torch.nn.functional as F

def _gaussian_blur_torch(heatmaps: torch.Tensor, kernel=11):
    """
    Simple Gaussian blur for each channel in [N, K, H, W].
    For brevity, no custom sigma. Adjust as you like.
    """
    # We'll do depthwise conv with groups=K*N or so, but let's keep it simple:
    # A quick way is to reshape: (N*K, 1, H, W) -> conv -> reshape back.
    # We skip dynamic sigma calculation. For a real "unbiased" approach,
    # you can compute sigma = 0.3*((kernel-1)*0.5 -1)+0.8 if needed.
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

def _get_max_preds_torch_soft(heatmaps: torch.Tensor, beta: float = 1.0):
    """
    Differentiable soft-argmax over each [N, K, H, W] channel.
    Returns:
      preds: (N, K, 2) Tensor (x, y) coordinates (float) computed as weighted average.
      maxvals: (N, K, 1) Tensor representing the maximum heatmap value per channel (for reference).
    """
    N, K, H, W = heatmaps.shape
    # Flatten heatmaps: shape (N, K, H*W)
    heatmaps_reshaped = heatmaps.view(N, K, -1)
    
    # Compute softmax over spatial locations with temperature scaling (beta)
    softmaxed = torch.softmax(beta * heatmaps_reshaped, dim=2)  # (N, K, H*W)
    softmaxed_2d = softmaxed.view(N, K, H, W)  # reshape back to 2D
    
    # Generate coordinate grids for x and y
    device = heatmaps.device
    # x: 0 ~ (W-1), y: 0 ~ (H-1)
    grid_x = torch.linspace(0, W - 1, W, device=device).view(1, 1, 1, W).expand(N, K, H, W)
    grid_y = torch.linspace(0, H - 1, H, device=device).view(1, 1, H, 1).expand(N, K, H, W)
    
    # Compute expected coordinates (weighted sum over x and y coordinates)
    pred_x = torch.sum(softmaxed_2d * grid_x, dim=(2, 3))
    pred_y = torch.sum(softmaxed_2d * grid_y, dim=(2, 3))
    
    preds = torch.stack([pred_x, pred_y], dim=2)  # (N, K, 2)
    
    # Option: Keep max values from the original hard max (for reference)
    maxvals, _ = torch.max(heatmaps_reshaped, dim=2, keepdim=True)
    maxvals = maxvals.view(N, K, 1)
    
    return preds, maxvals

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
    training=False
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

    # 3) Argmax => integer coords
    # if training:
    #     preds, maxvals = _get_max_preds_torch_soft(heatmaps, beta=25)  # use log-blurred for unbiased    
    # else:
    #     preds, maxvals = _get_max_preds_torch(heatmaps)  # use log-blurred for unbiased
    
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