import os
import sys
sys.path.append("./")
import os.path as osp
from glob import glob

import cv2
import torch
import hydra
import scipy.io
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf, DictConfig

torch.utils.data._utils.worker.IS_DAEMON = False
os.environ["PYTHONFAULTHANDLER"] = "1"
from loguru import logger

from runs.eval.inference import tools
from runs.eval.inference.registry import load_checkpoint, load_model

DATA_DIR = "/ps/project/datasets/LSPET_HR/images"
RESULTS_DIR = "/is/cluster/fast/sshin/results/inference/DeepGaitLab/LSPET_HR"

CUR_PATH = os.getcwd()
@hydra.main(version_base=None, config_path=os.path.join(CUR_PATH, "configs/train/models_2d"), config_name="config.yaml")
def main(cfg: DictConfig):
    
    OmegaConf.register_new_resolver("mult", lambda x,y: x*y)
    OmegaConf.register_new_resolver("if", lambda x, y, z: y if x else z)
    OmegaConf.register_new_resolver("div", lambda x, y: x // y)
    OmegaConf.register_new_resolver("concat", lambda x: np.concatenate(x))
    OmegaConf.register_new_resolver("sorted", lambda x: np.argsort(x))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(torch.cuda.get_device_properties(device))
    
    work_dir = osp.join(CUR_PATH, cfg.work_dir, 'train', 'regressor_2d', cfg.exp_name)
    ckpt_dir = os.path.join(work_dir, 'checkpoints')
    model = load_model(cfg, device)
    model = load_checkpoint(model, ckpt_dir)
    model.eval()

    annots = scipy.io.loadmat(os.path.join(DATA_DIR, "joints.mat"))
    
    joints_list = annots['joints'].transpose(2, 0, 1)
    image_pth_list = sorted(glob(os.path.join(DATA_DIR, "*.png")))

    assert len(joints_list) == len(image_pth_list)

    results_dir = os.path.join(RESULTS_DIR, cfg.exp_name)
    vis_dir = results_dir.replace("inference", "visualization")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    aspect_ratio = 4/3

    for joint, image_pth in (pbar := tqdm(zip(joints_list, image_pth_list), total=len(joints_list), dynamic_ncols=True)):
        cvimg = cv2.imread(image_pth)
        img_size = cvimg.shape[:2]
        xyxy, valid = tools.kp2xyxy(joint[None], img_size, aspect_ratio=aspect_ratio)

        if isinstance(valid, bool):
            continue

        center, scale = tools.xyxy2cs(xyxy[0], aspect_ratio, pixel_std=200.0, scale_factor=1.0)
        scale = scale.max()
        img_tensor = tools.process_cvimg(cvimg.copy(), center, scale, cfg.image_size)
        img_tensor = torch.from_numpy(img_tensor).float().to(device)

        with torch.no_grad():
            pred = model(img_tensor.unsqueeze(0))
        
        pred_joints2d = pred['joints2d'].cpu().squeeze(0).numpy()
        if pred_joints2d.shape[-1] == 2:
            pred_joints2d = np.concatenate((pred_joints2d, np.ones_like(pred_joints2d[..., :1])), axis=-1)
        
        pred_joints2d = tools.convert_kps_to_full_img(pred_joints2d, center, scale, cfg.image_size)

        # Visualize
        for xy in pred_joints2d:
            x = int(xy[0])
            y = int(xy[1])
            cv2.circle(cvimg, (x, y), radius=2, color=(0, 255, 0), thickness=-1)

        cv2.imwrite(os.path.join(vis_dir, os.path.basename(image_pth)), cvimg)


if __name__ == '__main__':
    main()