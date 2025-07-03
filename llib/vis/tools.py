import os
import cv2
import numpy as np
import torch
from PIL import Image

def get_colors():
    #     color_file = os.path.abspath(os.path.join(__file__, "../colors_phalp.txt"))
    color_file = os.path.abspath(os.path.join(__file__, "../colors.txt"))
    RGB_tuples = np.vstack(
        [
            np.loadtxt(color_file, skiprows=0),
            #             np.loadtxt(color_file, skiprows=1),
            np.random.uniform(0, 255, size=(10000, 3)),
            [[0, 0, 0]],
        ]
    )
    b = np.where(RGB_tuples == 0)
    RGB_tuples[b] = 1
    return RGB_tuples.astype(np.float32)

def checkerboard_geometry(
    length=12.0,
    color0=[0.8, 0.9, 0.9],
    color1=[0.6, 0.7, 0.7],
    tile_width=0.5,
    alpha=1.0,
    up="y",
    c1=0.0,
    c2=0.0,
):
    assert up == "y" or up == "z"
    color0 = np.array(color0 + [alpha])
    color1 = np.array(color1 + [alpha])
    radius = length / 2.0
    num_rows = num_cols = max(2, int(length / tile_width))
    vertices = []
    vert_colors = []
    faces = []
    face_colors = []
    for i in range(num_rows):
        for j in range(num_cols):
            u0, v0 = j * tile_width - radius, i * tile_width - radius
            us = np.array([u0, u0, u0 + tile_width, u0 + tile_width])
            vs = np.array([v0, v0 + tile_width, v0 + tile_width, v0])
            zs = np.zeros(4)
            if up == "y":
                cur_verts = np.stack([us, zs, vs], axis=-1)  # (4, 3)
                cur_verts[:, 0] += c1
                cur_verts[:, 2] += c2
            else:
                cur_verts = np.stack([us, vs, zs], axis=-1)  # (4, 3)
                cur_verts[:, 0] += c1
                cur_verts[:, 1] += c2

            cur_faces = np.array(
                [[0, 1, 3], [1, 2, 3], [0, 3, 1], [1, 3, 2]], dtype=np.int64
            )
            cur_faces += 4 * (i * num_cols + j)  # the number of previously added verts
            use_color0 = (i % 2 == 0 and j % 2 == 0) or (i % 2 == 1 and j % 2 == 1)
            cur_color = color0 if use_color0 else color1
            cur_colors = np.array([cur_color, cur_color, cur_color, cur_color])

            vertices.append(cur_verts)
            faces.append(cur_faces)
            vert_colors.append(cur_colors)
            face_colors.append(cur_colors)

    vertices = np.concatenate(vertices, axis=0).astype(np.float32)
    vert_colors = np.concatenate(vert_colors, axis=0).astype(np.float32)
    faces = np.concatenate(faces, axis=0).astype(np.float32)
    face_colors = np.concatenate(face_colors, axis=0).astype(np.float32)

    return vertices, faces, vert_colors, face_colors


def overlay_image_onto_background(image, mask, bbox, background):
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    out_image = background.copy()
    bbox = bbox[0].int().cpu().numpy().copy()
    roi_image = out_image[bbox[1]:bbox[3], bbox[0]:bbox[2]]

    roi_image[mask] = image[mask]
    out_image[bbox[1]:bbox[3], bbox[0]:bbox[2]] = roi_image

    return out_image


def update_intrinsics_from_bbox(K_org, bbox):
    device, dtype = K_org.device, K_org.dtype
    
    K = torch.zeros((K_org.shape[0], 4, 4)
    ).to(device=device, dtype=dtype)
    K[:, :3, :3] = K_org.clone()
    K[:, 2, 2] = 0
    K[:, 2, -1] = 1
    K[:, -1, 2] = 1
    
    image_sizes = []
    for idx, bbox in enumerate(bbox):
        left, upper, right, lower = bbox
        cx, cy = K[idx, 0, 2], K[idx, 1, 2]

        new_cx = cx - left
        new_cy = cy - upper
        new_height = max(lower - upper, 1)
        new_width = max(right - left, 1)
        new_cx = new_width - new_cx
        new_cy = new_height - new_cy

        K[idx, 0, 2] = new_cx
        K[idx, 1, 2] = new_cy
        image_sizes.append((int(new_height), int(new_width)))

    return K, image_sizes


def perspective_projection(x3d, K, R=None, T=None):
    if R != None:
        x3d = torch.matmul(R, x3d.transpose(1, 2)).transpose(1, 2)
    if T != None:
        x3d = x3d + T.transpose(1, 2)

    x2d = torch.div(x3d, x3d[..., 2:])
    x2d = torch.matmul(K, x2d.transpose(-1, -2)).transpose(-1, -2)[..., :2]
    return x2d


def compute_bbox_from_points(X, img_w, img_h, scaleFactor=1.2):
    left = torch.clamp(X.min(1)[0][:, 0], min=0, max=img_w)
    right = torch.clamp(X.max(1)[0][:, 0], min=0, max=img_w)
    top = torch.clamp(X.min(1)[0][:, 1], min=0, max=img_h)
    bottom = torch.clamp(X.max(1)[0][:, 1], min=0, max=img_h)

    cx = (left + right) / 2
    cy = (top + bottom) / 2
    width = (right - left)
    height = (bottom - top)

    new_left = torch.clamp(cx - width/2 * scaleFactor, min=0, max=img_w-1)
    new_right = torch.clamp(cx + width/2 * scaleFactor, min=1, max=img_w)
    new_top = torch.clamp(cy - height / 2 * scaleFactor, min=0, max=img_h-1)
    new_bottom = torch.clamp(cy + height / 2 * scaleFactor, min=1, max=img_h)

    bbox = torch.stack((new_left.detach(), new_top.detach(),
                        new_right.detach(), new_bottom.detach())).int().float().T
    
    return bbox