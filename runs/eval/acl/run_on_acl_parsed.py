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
from torch.utils.data import DataLoader
from hydra.core.hydra_config import HydraConfig

torch.utils.data._utils.worker.IS_DAEMON = False
os.environ["PYTHONFAULTHANDLER"] = "1"
from loguru import logger

from runs.eval.inference import tools
from runs.eval.inference.registry import load_checkpoint, load_model
from runs.eval.acl.utils import load_calibration, get_video_list
# from llib.datasets.IMOVE import IMOVEDataset

DATA_BASE_DIR = "/home/cmumbl/Data/ACL"
CALIB_PTH = f'{DATA_BASE_DIR}/calib/subject/time/calibration.txt'
RESULTS_DIR = f"{DATA_BASE_DIR}/DeepGaitLab_v02"
mean = np.array([0.485, 0.456, 0.406])[:, None, None]
std = np.array([0.229, 0.224, 0.225])[:, None, None]

CUR_PATH = os.getcwd()
@hydra.main(version_base=None, config_path=os.path.join(CUR_PATH, "configs/train/acl"), config_name="config.yaml")
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

    fldr = work_dir.split('/')[-2]
    save_fldr = '-'.join(cfg.exp_name.split('-')[1:])
    vis_fldr = 'vis_' + save_fldr
    save_dir = os.path.join(RESULTS_DIR, fldr, save_fldr)
    vis_dir = os.path.join(RESULTS_DIR, fldr, vis_fldr)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    # Data loading
    subject_list = sorted(os.listdir(osp.join(DATA_BASE_DIR, 'resampled_videos')))
    time_list = sorted(os.listdir(osp.join(DATA_BASE_DIR, 'resampled_videos', 'NIMBLE10')))
    # subject_list = [subject for subject in subject_list if not subject.startswith('.')]
    # activity_list = sorted(os.listdir(osp.join(DATA_BASE_DIR, 'resampled_videos', 'Subject_15', 'RGB_outputs')))
    # activity_list = [activity for activity in activity_list if not (activity.startswith('.'))]
    # subject_list = subject_list[cfg.demo.mod_part::cfg.demo.n_parts]
    for subject in ['NIMBLE10']:
    # for subject in subject_list:
        for time in time_list:
            os.makedirs(osp.join(vis_dir, subject, time), exist_ok=True)
            activity_list = sorted(os.listdir(osp.join(DATA_BASE_DIR, 'resampled_videos', subject, time, 'PTCTRC')))
            for activity in activity_list:
                os.makedirs(osp.join(save_dir, subject, time, activity), exist_ok=True)
                video_name_list, ext, camera_list = get_video_list(osp.join(DATA_BASE_DIR, 'resampled_videos', subject, time, 'PTCTRC', activity))
                Ks, Rs, Ts, dists, serials = load_calibration(CALIB_PTH.replace('subject', subject).replace('time', time), camera_list)
                import pdb; pdb.set_trace()
            
                parsed_label_pth = f"datasets/imove/imove_{subject}_{activity}.pkl"
                dataset = IMOVEDataset(
                    label_path=parsed_label_pth,
                    landmark_type="anatomy_v0",
                    normalize_plus_min_one=True,
                    fps=1,
                    is_multiview=True,
                )
                dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0, drop_last=False)
                
                save_pth_list = [osp.join(save_dir, subject, activity, camera + '.npy') for camera in camera_list]
                vis_pth = osp.join(vis_dir, subject, activity + '.mp4')

                # if all([osp.exists(save_pth) for save_pth in save_pth_list]):
                #     print(f'{subject} {activity} already processed')
                #     continue

                writer = imageio.get_writer(vis_pth, 
                                    fps=30, 
                                    format='FFMPEG', 
                                    mode='I', 
                                    quality=8, 
                                    macro_block_size=None
                )
                
                predictions = []
                for batch in tqdm(dataloader, desc="Processing batch", total=len(dataloader), leave=False, dynamic_ncols=True):
                    images = batch['image'].to(device)
                    masks = batch['mask'].to(device)
                    valid = batch['valid'].to(device)
                    bbox_info = batch['bbox_info'].to(device)
                    scale = batch['scale'].numpy()
                    center = batch['center'].numpy()
                    
                    with torch.no_grad():
                        pred = model._forward(x=images, masks=masks, bbox_info=bbox_info, valid=valid)

                    pred_joints2d = pred['joints2d'].cpu().numpy()
                    for i in range(pred["joints2d"].shape[0]):
                        for j in range(pred["joints2d"].shape[1]):
                            if not valid[i, j]:
                                pred_joints2d[i, j] = np.zeros_like(pred_joints2d[i, j])
                            else:
                                pred_joints2d[i, j] = tools.convert_kps_to_full_img(
                                    pred_joints2d[i, j], center[i, j], scale[i, j], cfg.image_size)
                    
                    if pred.get('visibility', None) is not None:
                        pred_joints2d = np.concatenate((
                            pred_joints2d, pred['visibility'].cpu().numpy()
                        ), axis=-1)
                    predictions.append(pred_joints2d)

                    if visualize:
                        for i in range(pred["joints2d"].shape[0]):
                            rows = 2
                            cols = 5
                            
                            full_img = np.zeros((rows * cfg.image_size[1], cols * cfg.image_size[0], 3), dtype=np.uint8)
                            for j in range(pred["joints2d"].shape[1]):
                                img = images[i, j].cpu().numpy()
                                img_norm = (img * std + mean).transpose(1, 2, 0)
                                img_norm_ori = (img_norm * 255).astype(np.uint8)
                                
                                row, col = j // cols, j % cols
                                crop_joints = tools.convert_kps_to_crop_img(
                                    pred_joints2d[i, j, :, :3].copy(), center[i, j], scale[i, j], cfg.image_size)
                                
                                for x, y in crop_joints[:, :2].astype(np.int32):
                                    img_norm_ori = cv2.circle(img_norm_ori, (x, y), 2, (0, 255, 0), -1)

                                full_img[row * cfg.image_size[1]:(row + 1) * cfg.image_size[1], 
                                        col * cfg.image_size[0]:(col + 1) * cfg.image_size[0]] = img_norm_ori
                            
                            writer.append_data(cv2.resize(full_img, None, fx=0.5, fy=0.5))
                writer.close()
                predictions = np.concatenate(predictions, axis=0).transpose(1, 0, 2, 3)
                for prediction, save_pth in zip(predictions, save_pth_list):
                    np.save(save_pth, prediction)

if __name__ == '__main__':
    main()