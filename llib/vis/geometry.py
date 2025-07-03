import torch
import numpy as np

from pytorch3d.renderer import (
    TexturesVertex,
)

from pytorch3d.utils import ico_sphere
from pytorch3d.structures import Meshes
from pytorch3d.structures.meshes import join_meshes_as_scene

def prep_shared_geometry(verts, faces, colors):
    """
    :param verts (V, 3) or (B, V, 3)
    :param faces (F, 3) or (B, F, 3)
    :param colors (B, 4) or (B, V, 4)
    """

    if len(verts.shape) == 2:
        verts = verts.unsqueeze(0)
    
    B, V = verts.shape[:2]
    
    F = faces.shape[-2]
    if len(faces.shape) == 2:
        faces = faces.unsqueeze(0)
    faces = faces.expand(B, F, -1)

    if len(colors.shape) == 2:
        colors = colors.unsqueeze(1)
    colors = colors.expand(B, V, -1)[..., :3]

    return verts, faces, colors


def create_meshes(verts, faces, colors):
    """
    :param verts (B, V, 3)
    :param faces (B, F, 3)
    :param colors (B, V, 3)
    """

    textures = TexturesVertex(verts_features=colors)
    meshes = Meshes(verts=verts, faces=faces, textures=textures)
    return join_meshes_as_scene(meshes)


def create_point_markers(position, radius):
    if isinstance(position, list):
        position = torch.from_numpy(position).float()
    
    device = position.device
    mesh = ico_sphere(level=0, device=device)
    
    verts = mesh.verts_list()[0] * radius
    faces = mesh.faces_list()[0]
    
    if len(position.shape) == 3:
        verts = verts.unsqueeze(0)
    elif len(position.shape) == 1:
        position = position.unsqueeze(0)
    
    verts = verts + position
    return verts, faces


def join_meshes(verts, faces, colors, renderer):
    mesh = renderer.join_multiple_meshes([verts], [faces], [colors])
    return mesh


def append_ground_geometry(verts, faces, colors, renderer):
    gv, gf, gc = renderer.ground_geometry
    mesh = renderer.join_multiple_meshes([verts, gv], [faces, gf], [colors, gc])

    return mesh


def process_mesh_group(verts, faces, colors):
    if isinstance(verts, list) or isinstance(verts, tuple):
        out_verts = [*verts]
    else:
        out_verts = [verts]
    if isinstance(faces, list) or isinstance(faces, tuple):
        out_faces = [*faces]
    else:
        out_faces = [faces]
    if (isinstance(colors, list) or isinstance(colors, tuple)):
        if len(colors) > 0 and isinstance(colors[0], float):
            out_colors = [colors]
        else:
            out_colors = [*colors]
    else:
        out_colors = [colors]
    return out_verts, out_faces, out_colors


def append_smpl_verts(verts, faces, colors, smpl_verts, smpl_faces, smpl_colors=(0.9, 0.9, 0.9)):
    """Add SMPL(X) vertices to the given mesh group"""
    out_verts, out_faces, out_colors = process_mesh_group(verts, faces, colors)

    if smpl_verts.ndim == 2:
        smpl_verts = smpl_verts.unsqueeze(0)
    elif smpl_verts.ndim == 4:
        smpl_verts = smpl_verts.unsqueeze(0)
    assert smpl_verts.ndim == 3
    
    if isinstance(smpl_faces, np.ndarray):
        smpl_faces = torch.from_numpy(smpl_faces.astype(np.int64)).to(smpl_verts.device)
    
    while smpl_faces.ndim > 2:
        smpl_faces = smpl_faces[0]

    for smpl_vert in smpl_verts:
        out_verts.append(smpl_vert)
        out_faces.append(smpl_faces)
        out_colors.append(smpl_colors)

    return out_verts, out_faces, out_colors


def append_point_markers(verts, faces, colors, position_list, radius=0.02, point_colors=(1.0, 0.3, 0.3)):
    """Add point markers to the given mesh group"""

    if isinstance(verts, list) or isinstance(verts, tuple):
        out_verts = [*verts]
    else:
        out_verts = [verts]
    if isinstance(faces, list) or isinstance(faces, tuple):
        out_faces = [*faces]
    else:
        out_faces = [faces]
    if (isinstance(colors, list) or isinstance(colors, tuple)):
        if len(colors) > 0 and isinstance(colors[0], float):
            out_colors = [colors]
        else:
            out_colors = [*colors]
    else:
        out_colors = [colors]
    
    for position in position_list:
        _v, _f = create_point_markers(position, radius)
        out_verts.append(_v)
        out_faces.append(_f)
        out_colors.append(point_colors)

    return out_verts, out_faces, out_colors