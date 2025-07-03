import os
import sys
sys.path.append("./")
import os.path as osp

from glob import glob

import cv2
import torch
import hydra

import imageio
from omegaconf import OmegaConf, DictConfig
from hydra.core.hydra_config import HydraConfig
import numpy as np
import tqdm

torch.utils.data._utils.worker.IS_DAEMON = False
os.environ["PYTHONFAULTHANDLER"] = "1"
import pytorch_lightning as pl
from loguru import logger

from configs import constants as _C
from runs.eval.inference.registry import load_checkpoint, load_model
from llib.trainer.diffusion.ldmks_prior import DiffusionTrainer as ModelCls

CUR_PATH = os.getcwd()
@hydra.main(version_base=None, config_path=os.path.join(CUR_PATH, "configs/train/models_2d"), config_name="config.yaml")
def main(cfg: DictConfig):
    visualize = True
    OmegaConf.register_new_resolver("mult", lambda x,y: x*y)
    OmegaConf.register_new_resolver("if", lambda x, y, z: y if x else z)
    OmegaConf.register_new_resolver("div", lambda x, y: x // y)
    OmegaConf.register_new_resolver("concat", lambda x: np.concatenate(x))
    OmegaConf.register_new_resolver("sorted", lambda x: np.argsort(x))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(torch.cuda.get_device_properties(device))
    
    config_search_path = HydraConfig.get().runtime.config_sources
    work_dir = config_search_path[1]['path']
    ckpt_dir = os.path.join(work_dir, 'checkpoints')
    model = load_model(cfg, device)
    model = load_checkpoint(model, ckpt_dir)
    model.eval()

    # Debugging, so specified experiment
    from runs.eval.imove.utils import load_calibration
    base_dir = "/fast/sshin/data/D10_data"
    subject = "Subject_10"
    activity = "t4_lat_step"
    # camera_names, camera_idxs = ['C11409', 'C11413'], [3, 7]
    camera_names, camera_idxs = ['C11398', 'C11399', 'C11400', 'C11409', 'C11410', 'C11411', 'C11412', 'C11413', 'C11414', 'C11415'], list(range(10))
    camera_names, camera_idxs = ['C11398', 'C11409', 'C11410', 'C11411', 'C11412', 'C11413', 'C11414', 'C11415'], [0, 3, 4, 5, 6, 7, 8, 9]
    calib_pth = os.path.join(base_dir, 'calibration', subject, "calibration.txt")
    Ks, Rs, Ts, dists, serials = load_calibration(calib_pth, camera_idxs)
    # ldmks_dir = os.path.join(base_dir, "DeepGaitLab_v02", "bedlamlab_2d", "ViT-Large-tune-mask-anatomy_v0", subject, activity)
    ldmks_dir = os.path.join(base_dir, "DeepGaitLab_v02", "bedlam", "ViT-Large-tune-mask-anatomy_v0", subject, activity)
    ldmks = np.stack([np.load(os.path.join(ldmks_dir, f"{camera}.npy")) for camera in camera_names], axis=1)
    import pdb; pdb.set_trace()
    
    # Apply triangulation for the initial frame
    from runs.eval.imove.triangulate import simple_triangulation
    from configs.landmarks import anatomy_v0 as ldmks_cfg
    ldmks3d = simple_triangulation(ldmks, {'Ks': Ks, 'Rs': Rs, 'Ts': Ts}, apply_conf=False)
    center = ldmks3d[:1, ldmks_cfg.center_idxs].mean(axis=1, keepdims=True)
    init_points = ldmks3d[:1]

    from llib.vis.renderer import Renderer
    from llib.vis.geometry import append_ground_geometry, append_point_markers
    vis_Ks, vis_Rs, vis_Ts, _, _ = load_calibration(calib_pth, [9])
    
    vis_Ks[0, 0, 0] /= 3
    vis_Ks[0, 1, 1] /= 3
    vis_Ks[0, 0, 2] /= 3
    vis_Ks[0, 1, 2] /= 3
    renderer = Renderer(width=640, height=360, focal_length=5000, K=vis_Ks[0], device=device)
    renderer.set_ground(length=10, center_x=0, center_z=0, up="z")
    renderer.cameras = renderer.create_camera(R=torch.from_numpy(vis_Rs[0]).cuda().float(), T=torch.from_numpy(vis_Ts[0]).cuda().float())
    
    writer = imageio.get_writer('test_prior.mp4', fps=30, mode='I', format='FFMPEG', macro_block_size=1)
    for start in tqdm.trange(1, ldmks3d.shape[0], cfg.data.train.window_size):
        if start + cfg.data.train.window_size > ldmks3d.shape[0]:
            break
        
        # repr_clean = torch.randn((1, cfg.data.train.window_size, cfg.model.encoder.n_landmarks, 3), dtype=torch.float32, device=device)
        # prompt = init_points - center
        
        # batch = dict(
        #     repr_clean=repr_clean,
        #     prompt=torch.from_numpy(prompt).float().to(device),
        #     ldmks2d=torch.from_numpy(ldmks[start:start+cfg.data.train.window_size]).float().to(device),
        #     Ks=torch.from_numpy(Ks).float().to(device),
        #     Rs=torch.from_numpy(Rs).float().to(device),
        #     Ts=torch.from_numpy(Ts).float().to(device),
        # )

        # pred_ldmks = model.network.inference(batch, grad_type='proj', cond_fn_with_grad=True)
        # pred_ldmks = (pred_ldmks.cpu().numpy() + center)[0]
        pred_ldmks = ldmks3d[start:start+cfg.data.train.window_size]
        for pred_ldmk in tqdm.tqdm(pred_ldmks, leave=False):
            _verts, _faces, _colors = append_point_markers(
                [], [], [], torch.from_numpy(pred_ldmk).cuda(), radius=0.02, point_colors=[0.5, 0.7, 1.0]
            )
            mesh = append_ground_geometry(_verts, _faces, _colors, renderer)

            img = renderer.render(mesh)
            # writer.append_data(cv2.resize(img, None, fx=0.5, fy=0.5))
            writer.append_data(img)

        center = pred_ldmks[-1:, ldmks_cfg.center_idxs].mean(axis=1, keepdims=True)
        init_points = pred_ldmks[-1:]
        
    writer.close()

        
    

if __name__ == '__main__':
    main()