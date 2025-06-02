# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

class JointGNLLLoss(nn.Module):
    """GNLL loss for SMPL-X parameters.

    Args:
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self, loss_weights=1.):
        super().__init__()
        self.criterion = nn.MSELoss()
        self.criterion_noreduce = nn.MSELoss(reduction='none')

        for name, weight in loss_weights.items():
            setattr(self, f'lw_{name}', weight)


    def forward(self, output, target, weights, with_sigma=False, kpts_loss_thresh=25):
        """Compute a (probabilistic) landmark/keypoint loss.

        Args:
            pred: Tensor shape (Batch, num_Keypoints * 3) if probabilistic and (Batch, num_Keypoints * 2) otherwise
            label_coords: Tensor shaped (Batch, num-Keypoints, 2) of keypoint coordinates
            weights: Tensor shaped (Batch, num-Keypoints) of per-keypoint weights
            with_sigma: Should probabilistic version of the loss be computed
            kpts_loss_thresh: Max value kpts_loss entries should take (set to very high if don't want thresholding)
        Returns:
            scalar loss
        """
        
        num_dims = output['joints2d'].shape[2]
        pred_coords = output['joints2d']

        kpts_diffs = target['joints'] - pred_coords[:, :, :2]  # shape: (B, K, 2)
        kpts_sq_diffs = torch.sum(torch.square(kpts_diffs), axis=-1)  # shape: (B, K)
        kpts_sq_diffs_weighted = torch.mul(kpts_sq_diffs, weights[:,:,0])
        
        if num_dims == 3:
            eps = torch.tensor(1e-6).to(pred_coords.device)
            pred_log_sigmas = pred_coords[:, :, -1]

            #clip sigmas
            pred_log_sigmas = torch.clip(pred_log_sigmas, min=torch.log(eps), max=None)
            pred_sigmas = torch.exp(pred_log_sigmas)

            keypoint_2_sigma_sq = 2.0 * torch.square(pred_sigmas)
            kpts_sq_diffs_over_2sigmasq = kpts_sq_diffs_weighted * (1.0 / keypoint_2_sigma_sq)

            # Clip kpts_loss to a maximum value as it can be very unstable due to (1.0 / keypoint_2_sigma_sq)
            kpts_loss = torch.mean(torch.clip(kpts_sq_diffs_over_2sigmasq, min=None, max=kpts_loss_thresh))
            sigmas_loss = torch.mean(2 * torch.mul(pred_log_sigmas, weights[:,:,0]))
            loss = self.lw_joints2d*(kpts_loss + sigmas_loss)

        else:
            kpts_loss = torch.mean(kpts_sq_diffs_weighted)
            loss = self.lw_joints2d*kpts_loss
            sigmas_loss = torch.tensor(0.0).to(kpts_loss.device)

        # total_loss = (loss_transl + loss_joints3d) * self.lw_total
        total_loss = dict(
            loss_joints2d = torch.mean(kpts_sq_diffs_weighted),
            loss = loss*self.lw_total,
            loss_sigma=sigmas_loss,
        )
        return total_loss


class KeypointsLoss(nn.Module):
    """Keypoints loss

    Args:
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self, loss_weights=1.):
        super().__init__()
        self.criterion = nn.MSELoss()
        self.criterion_noreduce = nn.MSELoss(reduction='none')

        for name, weight in loss_weights.items():
            setattr(self, f'lw_{name}', weight)


    def forward(self, pred_coords, target_coords, weights, kpts_loss_thresh=25):
        """Compute a (probabilistic) landmark/keypoint loss.

        Args:
            pred: Tensor shape (Batch, num_Keypoints * 3) if probabilistic and (Batch, num_Keypoints * 2) otherwise
            label_coords: Tensor shaped (Batch, num-Keypoints, 2) of keypoint coordinates
            weights: Tensor shaped (Batch, num-Keypoints) of per-keypoint weights
            with_sigma: Should probabilistic version of the loss be computed
            kpts_loss_thresh: Max value kpts_loss entries should take (set to very high if don't want thresholding)
        Returns:
            scalar loss
        """
        # pred_coords = output['joints3d'][:, :num_landmarks * 2].view(batch_size, num_landmarks, 2)

        kpts_diffs = target_coords - pred_coords[:, :, :2]  # shape: (B, K, 2)
        kpts_sq_diffs = torch.sum(torch.square(kpts_diffs), axis=-1)  # shape: (B, K)
        kpts_sq_diffs_weighted = torch.mul(kpts_sq_diffs, weights[:,:,0])
        # kpts_sq_diffs_weighted = kpts_sq_diffs

        kpts_loss = torch.mean(kpts_sq_diffs_weighted)
        loss = self.lw_joints2d*kpts_loss

        return loss * self.lw_total


class JointsMSELoss(nn.Module):
    """MSE loss for heatmaps.

    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self, use_target_weight=False, loss_weights=1.):
        super().__init__()
        self.criterion = nn.MSELoss()
        self.use_target_weight = use_target_weight
        self.loss_weights = loss_weights

    def forward(self, output, target, target_weight):
        """Forward function."""
        batch_size, num_joints, height, width = output.shape

        heatmaps_pred = output.reshape(
            (batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = 0.
        x = torch.linspace(-1, 1, width, device=output.device)
        y = torch.linspace(-1, 1, height, device=output.device)
        y, x = torch.meshgrid(y, x)
        y = y.reshape(-1)
        x = x.reshape(-1)

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze(1)
            heatmap_gt = heatmaps_gt[idx].squeeze(1)
            if self.use_target_weight:
                loss += self.criterion(heatmap_pred * target_weight[:, idx],
                                       heatmap_gt * target_weight[:, idx])
            else:
                loss += self.criterion(heatmap_pred, heatmap_gt)
            
        return loss / num_joints * self.loss_weights
