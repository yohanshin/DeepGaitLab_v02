import torch
import numpy as np

from pytorch3d.renderer import (
    TexturesVertex,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
)
from .geometry import create_meshes

class BaseRenderer():
    def __init__(self, width=None, height=None, focal_length=None, K=None, R=None, T=None, device=None, faces=None):

        self.width = width
        self.height = height
        self.focal_length = focal_length
        
        self.K = K
        self.R = R
        self.T = T

        self.device = device

        # Face definition
        if isinstance(faces, np.ndarray):
            F = faces.shape[-2]
            self.faces = torch.from_numpy(
                (faces).astype('int')
            ).reshape(1, F, 3).to(self.device)
        elif isinstance(faces, torch.Tensor):
            F = faces.shape[-2]
            self.faces = faces.reshape(1, F, 3).int().to(self.device)

        # Default camera
        self.register_cam_params()
    
    
    def create_renderer(self):
        """ Create base renderer """
        
        if hasattr(self, 'faces'):
            n_faces = len(self.faces[0])
        else:
            n_faces = 0
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                raster_settings=RasterizationSettings(
                    image_size=self.image_sizes[0],
                    blur_radius=1e-5,
                    max_faces_per_bin=int(max(20000, n_faces))),
            ),
            shader=SoftPhongShader(
                device=self.device,
                lights=self.lights,
            )
        )

    def register_cam_params(self):
        # Extrinsics
        if self.R is not None:
            if isinstance(self.R, np.ndarray):
                self.R = torch.from_numpy(self.R)
            self.R = self.R.reshape(1, 3, 3).float().to(self.device)
        else:
            self.R = torch.diag(
                torch.tensor([1, 1, 1])
            ).float().to(self.device).unsqueeze(0)

        if self.T is not None:
            if isinstance(self.T, np.ndarray):
                self.T = torch.from_numpy(self.T)
            self.T = self.T.reshape(1, 3).float().to(self.device)
        else:
            self.T = torch.tensor(
                [0, 0, 0]
            ).unsqueeze(0).float().to(self.device)

        # Intrinsics
        if self.K is not None:
            if isinstance(self.K, np.ndarray):
                self.K = torch.from_numpy(self.K)
            self.K = self.K.reshape(1, 3, 3).float().to(self.device)
        
        else:
            self.K = torch.tensor(
                [[self.focal_length, 0, self.width/2],
                [0, self.focal_length, self.height/2],
                [0, 0, 1]]
            ).unsqueeze(0).float().to(self.device)

    def prepare_textures(self, colors, vertices, return_texture=False):
        if len(vertices.shape) == 2:
            B = 1
            squeeze = True
        else:
            B = vertices.shape[0]
            squeeze = False

        if isinstance(colors, list) or isinstance(colors, tuple):
            if isinstance(colors, tuple): colors = list(colors)
            if colors[0] > 1: colors = [c / 255. for c in colors]
            colors = torch.tensor(colors).reshape(1, 1, 3).to(device=vertices.device, dtype=vertices.dtype)
        
        if colors.shape[-1] == 4:
            colors = colors[..., :3]
        
        colors = colors.reshape(B, -1, 3)
        colors = colors.expand(B, vertices.shape[-2], -1)

        if return_texture:
            return TexturesVertex(verts_features=colors)
        
        if squeeze:
            colors = colors.squeeze(0)
        return colors
    
    def join_multiple_meshes(self, verts_list, faces_list, colors_list):
        out_verts = []
        out_faces = []
        out_colors = []

        for verts, faces, colors in zip(verts_list, faces_list, colors_list):
            if isinstance(verts, torch.Tensor):

                if len(verts.shape) == 2:
                    verts = verts.unsqueeze(0)
                if len(faces.shape) == 2:
                    faces = faces.unsqueeze(0)
                colors = self.prepare_textures(colors, verts)
                
                out_verts += list(torch.unbind(verts, dim=0))
                out_faces += list(torch.unbind(faces, dim=0))
                out_colors += list(torch.unbind(colors, dim=0))

            elif isinstance(verts, list):
                colors = [self.prepare_textures(color, vert) for color, vert in zip(colors, verts)]
                
                out_verts += verts
                out_faces += faces
                out_colors += colors

        return create_meshes(out_verts, out_faces, out_colors)