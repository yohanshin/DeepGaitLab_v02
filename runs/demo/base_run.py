import os
import sys
sys.path.append("./")
import os.path as osp
from glob import glob

import cv2
import torch
import hydra
import imageio
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from omegaconf import OmegaConf, DictConfig
from hydra.core.hydra_config import HydraConfig

torch.utils.data._utils.worker.IS_DAEMON = False
os.environ["PYTHONFAULTHANDLER"] = "1"
from loguru import logger

from runs.eval.inference import tools
from runs.eval.inference.registry import load_checkpoint, load_model
from runs.eval.imove.utils import load_calibration, get_video_list

mean = np.array([0.485, 0.456, 0.406])[:, None, None]
std = np.array([0.229, 0.224, 0.225])[:, None, None]

CUR_PATH = os.getcwd()
@hydra.main(version_base=None, config_path=os.path.join(CUR_PATH, "configs/train/models_2d"), config_name="config.yaml")
def main(cfg: DictConfig):
    visualize = True
    OmegaConf.register_new_resolver("mult", lambda x,y: x*y)
    OmegaConf.register_new_resolver("if", lambda x, y, z: y if x else z)
    OmegaConf.register_new_resolver("div", lambda x, y: x // y)
    OmegaConf.register_new_resolver("concat", lambda x: np.concatenate(x))
    OmegaConf.register_new_resolver("sorted", lambda x: np.argsort(x))

    import pdb; pdb.set_trace()
    config_search_path = HydraConfig.get().runtime.config_sources

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(torch.cuda.get_device_properties(device))
    
    work_dir = config_search_path[1]['path']
    ckpt_dir = os.path.join(work_dir, 'checkpoints')
    model = load_model(cfg, device)
    model = load_checkpoint(model, ckpt_dir)
    model.eval()


if __name__ == "__main__":
    main()