import os
import sys
import argparse
from collections import defaultdict
sys.path.append("./")

import torch
import joblib
import numpy as np
from smplx import SMPLX
from scipy.spatial import cKDTree
import trimesh

from configs import constants as _C
from configs.landmarks import anatomy_v0, surface

def compute_vertex_normals(vertices, faces):
    """Compute vertex normals using face information."""
    # Initialize normals array
    normals = np.zeros_like(vertices)
    
    # Compute face normals
    face_normals = np.cross(
        vertices[faces[:, 1]] - vertices[faces[:, 0]],
        vertices[faces[:, 2]] - vertices[faces[:, 0]]
    )
    
    # Normalize face normals
    face_normals = face_normals / np.linalg.norm(face_normals, axis=1)[:, np.newaxis]
    
    # Accumulate face normals to vertices
    for i, face in enumerate(faces):
        normals[face] += face_normals[i]
    
    # Normalize vertex normals
    normals = normals / np.linalg.norm(normals, axis=1)[:, np.newaxis]
    
    return normals

def compute_face_normal(v1, v2, v3):
    """Compute normal of a face defined by three vertices."""
    normal = np.cross(v2 - v1, v3 - v1)
    norm = np.linalg.norm(normal)
    if norm > 0:
        normal = normal / norm
    return normal

def compute_triangle_angles(v1, v2, v3):
    """Compute the three angles of a triangle in radians."""
    # Compute edge vectors
    e1 = v2 - v1
    e2 = v3 - v2
    e3 = v1 - v3
    
    # Normalize edge vectors
    e1 = e1 / np.linalg.norm(e1)
    e2 = e2 / np.linalg.norm(e2)
    e3 = e3 / np.linalg.norm(e3)
    
    # Compute angles using dot products
    angle1 = np.arccos(np.clip(-np.dot(e1, e3), -1.0, 1.0))
    angle2 = np.arccos(np.clip(-np.dot(e1, e2), -1.0, 1.0))
    angle3 = np.arccos(np.clip(-np.dot(e2, e3), -1.0, 1.0))
    
    return np.array([angle1, angle2, angle3])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--landmarks', type=str, default='surface')
    args = parser.parse_args()

    if args.landmarks == "surface":
        cfg = surface
    elif args.landmarks == "anatomy_v0":
        cfg = anatomy_v0

    # Load the subset indices and get the corresponding vertices
    subset = joblib.load(cfg.subsample_pts_fn)
    subset = torch.argmax(subset, dim=-1).numpy()
    
    smplx = SMPLX(os.path.join(_C.PATHS.BODY_MODEL_DIR, "smplx"))
    
    # Create pose parameters
    batch_size = 1
    pose_params = torch.zeros(batch_size, smplx.NUM_BODY_JOINTS * 3, dtype=torch.float32)
    pose_params[0, 2] = 0.75
    pose_params[0, 5] = -0.75
    
    global_orient = torch.zeros(batch_size, 3, dtype=torch.float32)
    
    # Get vertices in X-pose
    output = smplx(
        global_orient=global_orient,
        body_pose=pose_params,
        return_verts=True,
        return_full_pose=True
    )
    original_vertices = output.vertices.detach().cpu().numpy()[0]  # Get first batch
    
    # Compute normals using original mesh
    original_normals = compute_vertex_normals(original_vertices, smplx.faces)
    
    # Get the positions and normals of our subsampled vertices
    subsampled_vertices = original_vertices[subset]
    subsampled_normals = original_normals[subset]
    
    # Build a KD-tree for fast nearest neighbor search
    tree = cKDTree(subsampled_vertices)
    
    # For each vertex, find its k nearest neighbors
    k = 12  # Number of nearest neighbors to consider
    criteria = 2.1
    distances, indices = tree.query(subsampled_vertices, k=k+1)  # k+1 because the first neighbor is the point itself
    
    # Create triangular faces
    new_faces = []
    for i in range(len(subsampled_vertices)):
        # Skip the first neighbor (it's the point itself)
        neighbors = indices[i][1:]
        
        # Create triangular faces with valid neighbors
        if len(neighbors) >= 2:
            # Create triangles with consecutive neighbors
            for j in range(len(neighbors)-1):
                v1 = neighbors[j]
                v2 = neighbors[j+1]
                
                # Compute face normal
                face_normal = compute_face_normal(
                    subsampled_vertices[i],
                    subsampled_vertices[v1],
                    subsampled_vertices[v2]
                )
                
                # Check if face normal is well-aligned with vertex normals
                # Use abs to handle flipped normals
                cosSum = np.abs(np.dot(face_normal, subsampled_normals[i])) + np.abs(np.dot(face_normal, subsampled_normals[v1])) + np.abs(np.dot(face_normal, subsampled_normals[v2]))
                
                # Compute triangle angles
                angles = compute_triangle_angles(
                    subsampled_vertices[i],
                    subsampled_vertices[v1],
                    subsampled_vertices[v2]
                )
                
                # Check if all angles are greater than 15 degrees (0.2618 radians)
                if cosSum > criteria and np.min(angles) > 0.2618:
                    new_faces.append([i, v1, v2])
    
    new_faces = np.array(new_faces)
    
    # Create Trimesh mesh
    mesh = trimesh.Trimesh(vertices=subsampled_vertices, faces=new_faces)
    
    # Save as PLY file
    output_dir = os.path.dirname(cfg.subsample_pts_fn)
    ply_path = "subsampled_mesh.ply"
    mesh.export(ply_path)
    print(f"Saved mesh to {ply_path}")
    import pdb; pdb.set_trace()
    
    # Also save the connectivity for later use
    output_path = os.path.join(output_dir, "smplx_to_dgl_faces.pkl")
    joblib.dump(new_faces, output_path)
    print(f"Saved connectivity to {output_path}")
    print(f"Number of new faces: {len(new_faces)}")
    