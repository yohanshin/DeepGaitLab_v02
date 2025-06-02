import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import math
import torch
import torchvision

from .top_down_eval import _get_max_preds, keypoints_from_heatmaps
# import ffmpeg


__all__ = ["joints_dict", "draw_points_and_skeleton"]


def joints_dict():
    joints = {
        "coco": {
            "keypoints": {
                0: "nose",
                1: "left_eye",
                2: "right_eye",
                3: "left_ear",
                4: "right_ear",
                5: "left_shoulder",
                6: "right_shoulder",
                7: "left_elbow",
                8: "right_elbow",
                9: "left_wrist",
                10: "right_wrist",
                11: "left_hip",
                12: "right_hip",
                13: "left_knee",
                14: "right_knee",
                15: "left_ankle",
                16: "right_ankle"
            },
            "skeleton": [
                # # [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8],
                # # [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]
                # [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7],
                # [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]
                [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7],
                [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4],  # [3, 5], [4, 6]
                [0, 5], [0, 6]
            ]
        },
        "mpii": {
            "keypoints": {
                0: "right_ankle",
                1: "right_knee",
                2: "right_hip",
                3: "left_hip",
                4: "left_knee",
                5: "left_ankle",
                6: "pelvis",
                7: "thorax",
                8: "upper_neck",
                9: "head top",
                10: "right_wrist",
                11: "right_elbow",
                12: "right_shoulder",
                13: "left_shoulder",
                14: "left_elbow",
                15: "left_wrist"
            },
            "skeleton": [
                # [5, 4], [4, 3], [0, 1], [1, 2], [3, 2], [13, 3], [12, 2], [13, 12], [13, 14],
                # [12, 11], [14, 15], [11, 10], # [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]
                [5, 4], [4, 3], [0, 1], [1, 2], [3, 2], [3, 6], [2, 6], [6, 7], [7, 8], [8, 9],
                [13, 7], [12, 7], [13, 14], [12, 11], [14, 15], [11, 10],
            ]
        },
    }
    return joints


def draw_points(image, points, color_palette='tab20', palette_samples=16, confidence_threshold=0.5):
    """
    Draws `points` on `image`.

    Args:
        image: image in opencv format
        points: list of points to be drawn.
            Shape: (nof_points, 3)
            Format: each point should contain (y, x, confidence)
        color_palette: name of a matplotlib color palette
            Default: 'tab20'
        palette_samples: number of different colors sampled from the `color_palette`
            Default: 16
        confidence_threshold: only points with a confidence higher than this threshold will be drawn. Range: [0, 1]
            Default: 0.5

    Returns:
        A new image with overlaid points

    """
    try:
        colors = np.round(
            np.array(plt.get_cmap(color_palette).colors) * 255
        ).astype(np.uint8)[:, ::-1].tolist()
    except AttributeError:  # if palette has not pre-defined colors
        colors = np.round(
            np.array(plt.get_cmap(color_palette)(np.linspace(0, 1, palette_samples))) * 255
        ).astype(np.uint8)[:, -2::-1].tolist()

    circle_size = max(1, min(image.shape[:2]) // 150)  # ToDo Shape it taking into account the size of the detection
    # circle_size = max(2, int(np.sqrt(np.max(np.max(points, axis=0) - np.min(points, axis=0)) // 16)))

    for i, pt in enumerate(points):
        if pt[2] > confidence_threshold:
            image = cv2.circle(image, (int(pt[1]), int(pt[0])), 3, tuple(colors[i % len(colors)]), -1)
            # image = cv2.circle(image, (int(pt[1]), int(pt[0])), circle_size, tuple(colors[i % len(colors)]), 2)

    return image


def draw_skeleton(image, points, skeleton, color_palette='Set2', palette_samples=8, person_index=0,
                  confidence_threshold=0.0):
    """
    Draws a `skeleton` on `image`.

    Args:
        image: image in opencv format
        points: list of points to be drawn.
            Shape: (nof_points, 3)
            Format: each point should contain (y, x, confidence)
        skeleton: list of joints to be drawn
            Shape: (nof_joints, 2)
            Format: each joint should contain (point_a, point_b) where `point_a` and `point_b` are an index in `points`
        color_palette: name of a matplotlib color palette
            Default: 'Set2'
        palette_samples: number of different colors sampled from the `color_palette`
            Default: 8
        person_index: index of the person in `image`
            Default: 0
        confidence_threshold: only points with a confidence higher than this threshold will be drawn. Range: [0, 1]
            Default: 0.5

    Returns:
        A new image with overlaid joints

    """
    try:
        colors = np.round(
            np.array(plt.get_cmap(color_palette).colors) * 255
        ).astype(np.uint8)[:, ::-1].tolist()
    except AttributeError:  # if palette has not pre-defined colors
        colors = np.round(
            np.array(plt.get_cmap(color_palette)(np.linspace(0, 1, palette_samples))) * 255
        ).astype(np.uint8)[:, -2::-1].tolist()

    for i, joint in enumerate(skeleton):
        pt1, pt2 = points[joint]
        if pt1[2] > confidence_threshold and pt2[2] > confidence_threshold:
            image = cv2.line(
                image, (int(pt1[1]), int(pt1[0])), (int(pt2[1]), int(pt2[0])),
                tuple(colors[person_index % len(colors)]), 2
            )

    return image


def draw_points_and_skeleton(image, points, skeleton, points_color_palette='tab20', points_palette_samples=16,
                             skeleton_color_palette='Set2', skeleton_palette_samples=8, person_index=0,
                             confidence_threshold=0.5):
    """
    Draws `points` and `skeleton` on `image`.

    Args:
        image: image in opencv format
        points: list of points to be drawn.
            Shape: (nof_points, 3)
            Format: each point should contain (y, x, confidence)
        skeleton: list of joints to be drawn
            Shape: (nof_joints, 2)
            Format: each joint should contain (point_a, point_b) where `point_a` and `point_b` are an index in `points`
        points_color_palette: name of a matplotlib color palette
            Default: 'tab20'
        points_palette_samples: number of different colors sampled from the `color_palette`
            Default: 16
        skeleton_color_palette: name of a matplotlib color palette
            Default: 'Set2'
        skeleton_palette_samples: number of different colors sampled from the `color_palette`
            Default: 8
        person_index: index of the person in `image`
            Default: 0
        confidence_threshold: only points with a confidence higher than this threshold will be drawn. Range: [0, 1]
            Default: 0.5

    Returns:
        A new image with overlaid joints

    """
    image = draw_skeleton(image, points, skeleton, color_palette=skeleton_color_palette,
                          palette_samples=skeleton_palette_samples, person_index=person_index,
                          confidence_threshold=confidence_threshold)
    image = draw_points(image, points, color_palette=points_color_palette, palette_samples=points_palette_samples,
                        confidence_threshold=confidence_threshold)
    return image


def save_images(images, target, joint_target, output, joint_output, joint_visibility, summary_writer=None, step=0,
                prefix=''):
    """
    Creates a grid of images with gt joints and a grid with predicted joints.
    This is a basic function for debugging purposes only.

    If summary_writer is not None, the grid will be written in that SummaryWriter with name "{prefix}_images" and
    "{prefix}_predictions".

    Args:
        images (torch.Tensor): a tensor of images with shape (batch x channels x height x width).
        target (torch.Tensor): a tensor of gt heatmaps with shape (batch x channels x height x width).
        joint_target (torch.Tensor): a tensor of gt joints with shape (batch x joints x 2).
        output (torch.Tensor): a tensor of predicted heatmaps with shape (batch x channels x height x width).
        joint_output (torch.Tensor): a tensor of predicted joints with shape (batch x joints x 2).
        joint_visibility (torch.Tensor): a tensor of joint visibility with shape (batch x joints).
        summary_writer (tb.SummaryWriter): a SummaryWriter where write the grids.
            Default: None
        step (int): summary_writer step.
            Default: 0
        prefix (str): summary_writer name prefix.
            Default: ""

    Returns:
        A pair of images which are built from torchvision.utils.make_grid
    """
    # Input images with gt
    images_ok = images.detach().clone()
    images_ok[:, 0].mul_(0.229).add_(0.485)
    images_ok[:, 1].mul_(0.224).add_(0.456)
    images_ok[:, 2].mul_(0.225).add_(0.406)
    for i in range(images.shape[0]):
        joints = joint_target[i] * 4.
        joints_vis = joint_visibility[i]

        for joint, joint_vis in zip(joints, joints_vis):
            if joint_vis[0]:
                a = int(joint[1].item())
                b = int(joint[0].item())
                # images_ok[i][:, a-1:a+1, b-1:b+1] = torch.tensor([1, 0, 0])
                images_ok[i][0, a - 1:a + 1, b - 1:b + 1] = 1
                images_ok[i][1:, a - 1:a + 1, b - 1:b + 1] = 0
    grid_gt = torchvision.utils.make_grid(images_ok, nrow=int(images_ok.shape[0] ** 0.5), padding=2, normalize=False)
    if summary_writer is not None:
        summary_writer.add_image(prefix + 'images', grid_gt, global_step=step)

    # Input images with prediction
    images_ok = images.detach().clone()
    images_ok[:, 0].mul_(0.229).add_(0.485)
    images_ok[:, 1].mul_(0.224).add_(0.456)
    images_ok[:, 2].mul_(0.225).add_(0.406)
    for i in range(images.shape[0]):
        joints = joint_output[i] * 4.
        joints_vis = joint_visibility[i]

        for joint, joint_vis in zip(joints, joints_vis):
            if joint_vis[0]:
                a = int(joint[1].item())
                b = int(joint[0].item())
                # images_ok[i][:, a-1:a+1, b-1:b+1] = torch.tensor([1, 0, 0])
                images_ok[i][0, a - 1:a + 1, b - 1:b + 1] = 1
                images_ok[i][1:, a - 1:a + 1, b - 1:b + 1] = 0
    grid_pred = torchvision.utils.make_grid(images_ok, nrow=int(images_ok.shape[0] ** 0.5), padding=2, normalize=False)
    if summary_writer is not None:
        summary_writer.add_image(prefix + 'predictions', grid_pred, global_step=step)

    # Heatmaps
    # ToDo
    # for h in range(0,17):
    #     heatmap = torchvision.utils.make_grid(output[h].detach(), nrow=int(np.sqrt(output.shape[0])),
    #                                            padding=2, normalize=True, range=(0, 1))
    #     summary_writer.add_image('train_heatmap_%d' % h, heatmap, global_step=step + epoch*len_dl_train)

    return grid_gt, grid_pred


def check_video_rotation(filename):
    # thanks to
    # https://stackoverflow.com/questions/53097092/frame-from-video-is-upside-down-after-extracting/55747773#55747773

    # this returns meta-data of the video file in form of a dictionary
    meta_dict = ffmpeg.probe(filename)

    # from the dictionary, meta_dict['streams'][0]['tags']['rotate'] is the key
    # we are looking for
    rotation_code = None
    try:
        if int(meta_dict['streams'][0]['tags']['rotate']) == 90:
            rotation_code = cv2.ROTATE_90_CLOCKWISE
        elif int(meta_dict['streams'][0]['tags']['rotate']) == 180:
            rotation_code = cv2.ROTATE_180
        elif int(meta_dict['streams'][0]['tags']['rotate']) == 270:
            rotation_code = cv2.ROTATE_90_COUNTERCLOCKWISE
        else:
            raise ValueError
    except KeyError:
        pass

    return rotation_code


def check_results(images, outputs, iteration, exp_name):
    import os
    os.makedirs(f'experiments/{exp_name}/visualization', exist_ok=True)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Unnormalize images
    images = images.detach().cpu()
    images = images * torch.tensor(std).view(1, 3, 1, 1) + torch.tensor(mean).view(1, 3, 1, 1)
    images = (images * 255).numpy().astype(np.uint8).transpose(0, 2, 3, 1)[..., ::-1]

    points, confs = _get_max_preds(outputs.detach().cpu().numpy())
    points = points[..., ::-1] * 4
    points = np.concatenate((points, confs), axis=-1)

    writer = imageio.get_writer(f"experiments/{exp_name}/visualization/results_{iteration}.mp4", fps=10)
    for image, point in zip(images, points):
        image = draw_points(image.copy(), point)
        writer.append_data(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    writer.close()


def check_heatmaps(images, outputs, iteration, exp_name):
    import os
    os.makedirs(f'experiments/{exp_name}/visualization', exist_ok=True)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Unnormalize images
    images = images.detach().cpu()
    images = images * torch.tensor(std).view(1, 3, 1, 1) + torch.tensor(mean).view(1, 3, 1, 1)
    images = (images * 255).numpy().astype(np.uint8).transpose(0, 2, 3, 1)[..., ::-1]

    outputs = outputs.detach().cpu().numpy()

    writer = imageio.get_writer(f"experiments/{exp_name}/visualization/results_{iteration}.mp4", fps=10)
    for image, output in zip(images, outputs):
        output = output.sum(0)
        output = np.clip(output * 255, 0, 255)
        output = output.astype(np.uint8)
        output = cv2.resize(output, (image.shape[1], image.shape[0]))
        heatmap = cv2.applyColorMap(output, cv2.COLORMAP_JET)
        super_image = cv2.addWeighted(heatmap, 0.5, image.copy(), 0.5, 0)

        writer.append_data(cv2.cvtColor(super_image, cv2.COLOR_BGR2RGB))

    writer.close()


def draw_heatmap(image, heatmap):
    output = np.clip(heatmap.sum(0) * 255, 0, 255)
    output = output.astype(np.uint8)
    output = cv2.resize(output, (image.shape[1], image.shape[0]))
    heatmap = cv2.applyColorMap(output, cv2.COLORMAP_JET)
    super_image = cv2.addWeighted(heatmap, 0.5, image.copy(), 0.5, 0)
    return super_image


def compare_results(images, preds, targets, iteration, path):
    import os

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Unnormalize images
    images = images.detach().cpu()
    images = images * torch.tensor(std).view(1, 3, 1, 1) + torch.tensor(mean).view(1, 3, 1, 1)
    images = (images * 255).numpy().astype(np.uint8).transpose(0, 2, 3, 1)[..., ::-1]

    h, w = images.shape[1:3]

    writer = imageio.get_writer(f"{path}/results_{iteration}.mp4", fps=10)
    for image, pred, target in zip(images, preds, targets):
        pred_points, prob = keypoints_from_heatmaps(
            heatmaps=pred[None], center=np.array([[w//2, h//2]]), scale=np.array([[w, h]]),
            unbiased=True, use_udp=True, kernel=11)

        target_points, prob = keypoints_from_heatmaps(
            heatmaps=target[None], center=np.array([[w//2, h//2]]), scale=np.array([[w, h]]),
            unbiased=True, use_udp=True, kernel=11)

        pred_image = image.copy()
        target_image = image.copy()

        for pred_point, target_point in zip(pred_points[0], target_points[0]):
            pred_image = cv2.circle(pred_image, (int(pred_point[0]), int(pred_point[1])), 3, (0, 255, 0), -1)
            target_image = cv2.circle(target_image, (int(target_point[0]), int(target_point[1])), 3, (0, 255, 0), -1)

        image = np.concatenate((image, pred_image, target_image), axis=1)

        writer.append_data(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    writer.close()

def compare_results_denseldmks2d(images, pred_vals, targets, masks, iteration, path, normalize_plus_min_one=False):
    import os

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Unnormalize images
    images = images.detach().cpu()
    images = images * torch.tensor(std).view(1, 3, 1, 1) + torch.tensor(mean).view(1, 3, 1, 1)
    images = (images * 255).numpy().astype(np.uint8).transpose(0, 2, 3, 1)[..., ::-1]
    masks = (masks*255).detach().cpu().numpy().astype(np.uint8).transpose(0, 2, 3, 1)[..., ::-1] if masks is not None else [None]*len(images)
    preds = pred_vals["joints2d"].detach().cpu().numpy()
    visibility = torch.sigmoid(pred_vals["visibility"]).squeeze(-1).detach().cpu().numpy() if "visibility" in pred_vals else None
    visibility_gt = targets["joints_visibility"].squeeze(-1).detach().cpu().numpy() if "visibility" in pred_vals else None

    # breakpoint()
    if len(preds.shape) == 3:
        preds[:, :, -1] = np.exp(preds[:, :, -1])
        preds[:, :, -1] = preds[:, :, -1]/(preds[:, :, -1].max())
    h, w = images.shape[1:3]
    video_fn = f"{path}/results_{iteration:010}.mp4"
    writer = imageio.get_writer(video_fn, fps=10)
    for i, (image, pred, target, mask) in enumerate(zip(images, preds, targets["joints"], masks)):

        pred_image = image.copy()
        target_image = image.copy()
        vis_image = image.copy()
        gt_vis_image = image.copy()
        mask_image = None

        # name images
        # pred_image = cv2.putText(pred_image, "Pred ldmks w/ uncert. (0-50 pxls)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        pred_img_txt = "Pred ldmks w/ uncert. in pxls: 0 to "
        pred_image = cv2.putText(pred_image, pred_img_txt,
                         (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        text_size, _ = cv2.getTextSize(pred_img_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        text_width = text_size[0]
        # Draw "50" in red right after the first text
        pred_image = cv2.putText(pred_image, "50",
                                (10 + text_width, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        if mask is not None:
            mask_image = np.repeat(mask, 3, axis=-1)
            mask_image = cv2.putText(mask_image, "Mask", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        target_image = cv2.putText(target_image, "GT ldmks", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        vis_image = cv2.putText(vis_image, "Visibile ldmks abs error", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        gt_vis_image = cv2.putText(gt_vis_image, "Not visible ldmks abs error", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        for j, (pred_point, target_point) in enumerate(zip(pred, target)):
            if pred_point.shape[-1] == 3:
                pred_x, pred_y, pred_sigma = pred_point
                normalize_contstant = 2 if normalize_plus_min_one else 1
                pred_sigma = math.sqrt(pred_sigma) / normalize_contstant * max(pred_image.shape[:2])
                max_uncertainty = 50  # pixels
                pred_sigma = pred_sigma/max_uncertainty
                pred_sigma = np.clip(pred_sigma, 0, 1)
            else:
                pred_x, pred_y = pred_point
                pred_sigma = 0.
            if normalize_plus_min_one:
                pred_x = (pred_x + 1) / 2
                pred_y = (pred_y + 1) / 2
                target_point = (target_point + 1) / 2

            pred_x = pred_x * w
            pred_y = pred_y * h
            circle_size = int(1*h//256)
            pred_image = cv2.circle(pred_image, (int(pred_x), int(pred_y)), circle_size, (0, int(255*(1-pred_sigma)), int(255*pred_sigma)), -1)
            target_x, target_y = target_point
            target_x = target_x * w
            target_y = target_y * h
            target_image = cv2.circle(target_image, (int(target_x), int(target_y)), circle_size, (0, 255, 0), -1)
            if visibility is not None:
                vis_diff = np.abs(visibility[i,j] - visibility_gt[i,j])
                if visibility_gt[i,j] > 0.5:
                    # green visible / red not visible
                    vis_image = cv2.circle(vis_image, (int(target_x), int(target_y)), circle_size,
                                        (0, int(255*(1-vis_diff)), int(255*vis_diff)), -1)
                else:
                    gt_vis_image = cv2.circle(gt_vis_image, (int(target_x), int(target_y)), circle_size,
                                            (0, int(255*(1-vis_diff)), int(255*vis_diff)), -1)
        if visibility is not None:
            image = np.concatenate((image, target_image, pred_image, vis_image, gt_vis_image), axis=1)
        else:
            image = np.concatenate((image, target_image, pred_image), axis=1)
        if mask is not None:
            image = np.concatenate((image, mask_image), axis=1)

        writer.append_data(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    writer.close()

    return video_fn


def compare_results_denseldmks2d(images, pred_vals, targets, iteration, path, normalize_plus_min_one=False):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Unnormalize images
    images = images.detach().cpu()
    images = images * torch.tensor(std).view(1, 3, 1, 1) + torch.tensor(mean).view(1, 3, 1, 1)
    images = (images * 255).numpy().astype(np.uint8).transpose(0, 2, 3, 1)[..., ::-1]
    preds = pred_vals["joints2d"].detach().cpu().numpy()
    
    has_visibility = "visibility" in pred_vals and pred_vals["visibility"] is not None
    visibility = torch.sigmoid(pred_vals["visibility"]).squeeze(-1).detach().cpu().numpy() if has_visibility else None
    visibility_gt = targets["joints_visibility"].squeeze(-1).detach().cpu().numpy() if has_visibility else None

    # breakpoint()
    if len(preds.shape) == 3 and preds.shape[-1] == 3:
        preds[:, :, -1] = np.exp(preds[:, :, -1])
        preds[:, :, -1] = preds[:, :, -1]/(preds[:, :, -1].max())
    h, w = images.shape[1:3]
    video_fn = f"{path}/results_{iteration:010}.mp4"
    writer = imageio.get_writer(video_fn, fps=10)
    for i, (image, pred, target) in enumerate(zip(images, preds, targets["joints"])):

        pred_image = image.copy()
        target_image = image.copy()
        vis_image = image.copy()
        gt_vis_image = image.copy()
        
        # name images
        # pred_image = cv2.putText(pred_image, "Pred ldmks w/ uncert. (0-50 pxls)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        pred_img_txt = "Pred ldmks w/ uncert. in pxls: 0 to "
        pred_image = cv2.putText(pred_image, pred_img_txt,
                         (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        text_size, _ = cv2.getTextSize(pred_img_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        text_width = text_size[0]
        # Draw "50" in red right after the first text
        pred_image = cv2.putText(pred_image, "50",
                                (10 + text_width, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        target_image = cv2.putText(target_image, "GT ldmks", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        vis_image = cv2.putText(vis_image, "Visibile ldmks abs error", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        gt_vis_image = cv2.putText(gt_vis_image, "Not visible ldmks abs error", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        for j, (pred_point, target_point) in enumerate(zip(pred, target)):
            if np.isnan(pred_point).any(): continue
            if pred_point.shape[-1] == 3:
                pred_x, pred_y, pred_sigma = pred_point
                normalize_contstant = 2 if normalize_plus_min_one else 1
                pred_sigma = math.sqrt(pred_sigma) / normalize_contstant * max(pred_image.shape[:2])
                max_uncertainty = 50  # pixels
                pred_sigma = pred_sigma/max_uncertainty
                pred_sigma = np.clip(pred_sigma, 0, 1)
            else:
                pred_x, pred_y = pred_point
                pred_sigma = 0.
            if normalize_plus_min_one:
                pred_x = (pred_x + 1) / 2
                pred_y = (pred_y + 1) / 2
                target_point = (target_point + 1) / 2

            pred_x = pred_x * w
            pred_y = pred_y * h
            circle_size = int(1*h//256)
            pred_image = cv2.circle(pred_image, (int(pred_x), int(pred_y)), circle_size, (0, int(255*(1-pred_sigma)), int(255*pred_sigma)), -1)
            target_x, target_y = target_point
            target_x = target_x * w
            target_y = target_y * h
            target_image = cv2.circle(target_image, (int(target_x), int(target_y)), circle_size, (0, 255, 0), -1)
            if visibility is not None:
                vis_diff = np.abs(visibility[i,j] - visibility_gt[i,j])
                if visibility_gt[i,j] > 0.5:
                    # green visible / red not visible
                    vis_image = cv2.circle(vis_image, (int(target_x), int(target_y)), circle_size,
                                        (0, int(255*(1-vis_diff)), int(255*vis_diff)), -1)
                else:
                    gt_vis_image = cv2.circle(gt_vis_image, (int(target_x), int(target_y)), circle_size,
                                            (0, int(255*(1-vis_diff)), int(255*vis_diff)), -1)
        if visibility is not None:
            image = np.concatenate((image, target_image, pred_image, vis_image, gt_vis_image), axis=1)
        else:
            image = np.concatenate((image, target_image, pred_image), axis=1)
        
        writer.append_data(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    writer.close()

    return video_fn