import cv2
import torch
import numpy as np

from pytorch3d.renderer import (
    PerspectiveCameras,
    TexturesVertex,
    PointLights,
    Materials,
)
from pytorch3d.structures import Meshes
from pytorch3d.structures.meshes import join_meshes_as_scene

from .base_renderer import BaseRenderer
from .tools import (checkerboard_geometry, 
                    update_intrinsics_from_bbox, 
                    perspective_projection,
                    overlay_image_onto_background,
                    compute_bbox_from_points
)


class Renderer(BaseRenderer):
    def __init__(self, width, height, focal_length=None, K=None, R=None, T=None, device=None, faces=None):
        super().__init__(width, height, focal_length, K, R, T, device, faces)

        self.initialize_camera_params()
        self.create_lights(location=[[0.0, 0.0, -10.0]])
        self.create_renderer()


    def create_lights(self, location=[[0.0, 0.0, -10.0]]):
        # Location is in camera coordinate, move it to the world coordinate
        if isinstance(location, list):
            location = torch.tensor(location).float().to(self.device)
        
        location = (location - self.T).reshape(1, 3, 1)
        location = (self.R.mT @ location).reshape(1, 3)
        self.lights = PointLights(device=self.device, location=location)

    
    def create_camera(self, R=None, T=None):
        if R is not None:
            self.R = R.clone().view(1, 3, 3).to(self.device)
        if T is not None:
            self.T = T.clone().view(1, 3).to(self.device)

        return PerspectiveCameras(
            device=self.device,
            R=self.R.mT,
            T=self.T,
            K=self.K_full,
            image_size=self.image_sizes,
            in_ndc=False)


    def initialize_camera_params(self):
        """Hard coding for camera parameters
        TODO: Do some soft coding"""

        self.bboxes = torch.tensor([[0, 0, self.width, self.height]]).float()
        self.K_full, self.image_sizes = update_intrinsics_from_bbox(self.K, self.bboxes)
        self.cameras = self.create_camera()
        
        
    def set_ground(self, length, center_x, center_z, up="y"):
        device = self.device
        v, f, vc, fc = map(torch.from_numpy, checkerboard_geometry(length=length, c1=center_x, c2=center_z, up=up))
        v, f, vc = v.to(device), f.to(device), vc.to(device)
        self.ground_geometry = [v, f, vc]


    def update_bbox(self, x3d, scale=2.0, mask=None):
        """ Update bbox of cameras from the given 3d points

        x3d: input 3D keypoints (or vertices), (num_frames, num_points, 3)
        """

        if x3d.size(-1) != 3:
            x2d = x3d.unsqueeze(0)
        else:
            x2d = perspective_projection(x3d.unsqueeze(0), self.K, self.R, self.T.reshape(1, 3, 1))

        if mask is not None:
            x2d = x2d[:, ~mask]

        bbox = compute_bbox_from_points(x2d, self.width, self.height, scale)
        self.bboxes = bbox

        self.K_full, self.image_sizes = update_intrinsics_from_bbox(self.K, bbox)
        self.cameras = self.create_camera()
        self.create_renderer()

    def reset_bbox(self,):
        bbox = torch.zeros((1, 4)).float().to(self.device)
        bbox[0, 2] = self.width
        bbox[0, 3] = self.height
        self.bboxes = bbox

        self.K_full, self.image_sizes = update_intrinsics_from_bbox(self.K, bbox)
        self.cameras = self.create_camera()
        self.create_renderer()


    def render(self, mesh, background=None):
        
        materials = Materials(
            device=self.device,
            shininess=0
        )

        results = torch.flip(
            self.renderer(mesh, materials=materials, cameras=self.cameras, lights=self.lights),
            [1, 2]
        )
        image = (results[0, ..., :3].cpu().numpy() * 255).astype(np.uint8)

        if background is not None:
            mask = results[0, ..., -1] > 1e-3
            image = overlay_image_onto_background(image, mask, self.bboxes, background.copy())
            
        return image


    def render_mesh(self, vertices, background, colors=[0.8, 0.8, 0.8]):
        self.update_bbox(vertices[::50], scale=1.2)
        vertices = vertices.unsqueeze(0)
        
        textures = self.prepare_textures(colors, vertices)
        mesh = Meshes(verts=vertices,
                      faces=self.faces,
                      textures=textures,)
        
        return self.render(mesh, background)
        
        
    def render_with_ground(self, verts, faces, colors, background=None):
        """
        :param verts (B, V, 3)
        :param faces (F, 3)
        :param colors (B, 3)
        """
        
        verts, faces, colors = prep_shared_geometry(verts, faces, colors)
        
        gv, gf, gc = self.ground_geometry
        mesh = self.join_multiple_meshes([verts, gv], [faces, gf], [colors, gc])
        return self.render(mesh, background)


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