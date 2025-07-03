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
from omegaconf import OmegaConf, DictConfig
from hydra.core.hydra_config import HydraConfig

torch.utils.data._utils.worker.IS_DAEMON = False
os.environ["PYTHONFAULTHANDLER"] = "1"
from loguru import logger

from runs.eval.inference import tools
from runs.eval.inference.registry import load_checkpoint, load_model
from runs.eval.acl.utils import load_calibration, get_video_list

DATA_BASE_DIR = "/home/cmumbl/Data/ACL"
CALIB_PTH = f'{DATA_BASE_DIR}/calib/subject/time/calibration.txt'
RESULTS_DIR = f"{DATA_BASE_DIR}/DeepGaitLab_v02"


def inference(cfg,
              vidcap_list, 
              bbox_pth_list, 
              mask_dir_list,
              Ks, 
              model, 
              save_pth_list, 
              vis_pth_list,
              device, 
              visualize,):
    
    mask_seqs = []
    for mask_pth in mask_dir_list:
        # mmap_mode='r' will not load the data into memory, but will allow you to access it
        npz = np.load(mask_pth, mmap_mode='r')
        key = npz.files[0]
        seq = npz[key] 
        mask_seqs.append(seq)

    
    image_size = cfg.image_size
    bbox_list = [np.load(bbox_pth)[::3] for bbox_pth in bbox_pth_list]
    
    max_vid_frames = min([int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)) for vidcap in vidcap_list])
    max_bbox_frames = min([bbox.shape[0] for bbox in bbox_list])

    max_frames = min(max_vid_frames, max_bbox_frames)
    bar = tqdm(range(max_frames), desc='Detecting', dynamic_ncols=True, leave=False)
    
    if visualize:
        writer_list = [
            imageio.get_writer(vis_pth, 
                               fps=30, 
                               format='FFMPEG', 
                               mode='I', 
                               quality=8, 
                               macro_block_size=None
            ) for vis_pth in vis_pth_list]
    
    frame_i = 0
    predictions = []
    while frame_i < max_frames:
        imgs, valids = [], []
        for vidcap in vidcap_list:
            ret, img = vidcap.read()
            imgs.append(img)
            valids.append(ret)
    
        # Check if all frames are valid
        if not all(valids):
            break
        
        valid_mask, img_tensors, centers, scales, masks = [], [], [], [], []
        for view_i, (frame, bbox, K) in enumerate(zip(imgs, bbox_list, Ks)):
            if np.all(bbox[frame_i] == 0.0):
                img_tensors.append(torch.zeros((1, 3, image_size[1], image_size[0])).float().to(device))
                valid_mask.append(False)
                centers.append(np.zeros(2))
                scales.append(np.zeros(2))
                masks.append(torch.zeros((1, 1, image_size[1], image_size[0])).float().to(device))
                continue
        
            cvimg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            center, scale = tools.xyxy2cs(bbox[frame_i], 0.75, pixel_std=200.0)
            scale = scale.max()
            img_tensor = tools.process_cvimg(cvimg.copy(), center, scale, image_size)
            img_tensor = torch.from_numpy(img_tensor).float().unsqueeze(0).to(device)
            img_tensors.append(img_tensor)
            centers.append(center)
            scales.append(scale)
            valid_mask.append(True)
            
            if model.use_mask:
                # mask_pth = os.path.join(mask_dir_list[view_i], f"{frame_i:05}.npy")
                # mask = np.load(mask_pth).astype(np.float32)
                # mask_npz = np.load(mask_dir_list[view_i])
                # mask = mask_npz[mask_npz.files[0]][frame_i].astype(np.float32) 
                mask = mask_seqs[view_i][frame_i].astype(np.float32)
                mask = tools.process_cvimg(mask.copy(), center, scale, image_size, transform_only=True)
                mask = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0).to(device)
                masks.append(mask)
            
        with torch.no_grad():
            if model.use_mask:
                pred = model(torch.cat(img_tensors), masks=torch.cat(masks))
            else:
                pred = model(torch.cat(img_tensors), masks=None)

        pred_joints2d = pred['joints2d'].cpu().numpy()
        for view_i, (pred_joint2d, center, scale, valid) in enumerate(zip(pred_joints2d, centers, scales, valid_mask)):
            if not valid:
                pred_joints2d[view_i] = np.zeros_like(pred_joints2d[view_i])
            else:
                pred_joints2d[view_i] = tools.convert_kps_to_full_img(pred_joint2d, center, scale, cfg.image_size)
        if "visibility" in pred.keys():
            pred_joints2d = np.concatenate((
                pred_joints2d, pred['visibility'].cpu().numpy()
            ), axis=-1)
        predictions.append(pred_joints2d)

        if visualize:
            for img, writer, pred_joint2d, center, scale in zip(imgs, writer_list, pred_joints2d, centers, scales):
                outimg = img.copy()
                outimg = tools.draw_points(cfg, pred_joint2d, outimg, scale=scale * 200)
                outimg = tools.process_cvimg(outimg, center, scale, image_size, transform_only=True)
                
                writer.append_data(cv2.cvtColor(outimg, cv2.COLOR_BGR2RGB))
        
        frame_i += 1
        bar.update(1)

    predictions = np.stack(predictions, axis=1)
    # Unnormalize
    predictions[..., :2] = (predictions[..., :2] + 1) / 2.0
    predictions[..., 0] *= image_size[0]
    predictions[..., 1] *= image_size[1]
    
    for prediction, save_pth in zip(predictions, save_pth_list):
        np.save(save_pth, prediction)
        


CUR_PATH = os.getcwd()
@hydra.main(version_base=None, config_path=os.path.join(CUR_PATH, "configs/train/acl"), config_name="config.yaml")
def main(cfg: DictConfig):
    
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

    tags = cfg.exp_name.split('-')
    save_fldr = os.path.join(tags[0], '-'.join(tags[1:-5])).replace("_2025", "")
    vis_fldr = os.path.join(tags[0], 'vis_' + '-'.join(tags[1:-7]))
    save_dir = os.path.join(RESULTS_DIR, save_fldr)
    vis_dir = os.path.join(RESULTS_DIR, vis_fldr)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    import pdb; pdb.set_trace()
    # Data loading
    subject_list = sorted(os.listdir(osp.join(DATA_BASE_DIR, 'resampled_videos')))
    time_list = sorted(os.listdir(osp.join(DATA_BASE_DIR, 'resampled_videos', 'NIMBLE10')))
    
    for subject in ['NIMBLE10']:
    # for subject in subject_list:
        for time in time_list:
            activity_list = sorted(os.listdir(osp.join(DATA_BASE_DIR, 'resampled_videos', subject, time, 'PTCTRC')))
            activity_list = ['Static1', 'Static2', 'LateralStepDown1', ] 
            for activity in activity_list:
                save_activity_dir = osp.join(save_dir, subject, time, activity)
                vis_activity_dir  = osp.join(vis_dir,  subject, time, activity)
                os.makedirs(save_activity_dir, exist_ok=True)
                os.makedirs(vis_activity_dir,  exist_ok=True)

                n_saved = len(os.listdir(save_activity_dir))
                n_vis   = len(os.listdir(vis_activity_dir))
                if n_saved ==6 and n_vis == 6:
                    print(f"[skip]{subject} {time} {activity} already processed.")
                    continue

                video_name_list, ext, camera_list = get_video_list(osp.join(DATA_BASE_DIR, 'resampled_videos', subject, time, 'PTCTRC', activity))
                Ks, Rs, Ts, dists, serials = load_calibration(CALIB_PTH.replace('subject', subject).replace('time', time), camera_list)
            
                video_dir = osp.join(DATA_BASE_DIR, 'resampled_videos', subject, time, 'PTCTRC', activity)
                bbox_dir = osp.join(DATA_BASE_DIR, 'bboxes', f'BBoxes_{time}', subject, 'Refined_YOLO_Detection', activity)
                mask_base_dir = os.path.join(DATA_BASE_DIR, "samurai_results", "masks", subject, f'Masks_{time}', activity)
                
                vidcap_list, bbox_pth_list, mask_dir_list = [], [], []
                for i, camera in enumerate(camera_list):
                    if camera in ["3177649"]: 
                        bbox_pth = glob(osp.join(bbox_dir, f'Rotated*_{camera}_*' + '.npy'))[0]
                    else:
                        bbox_pth = glob(osp.join(bbox_dir, f'DSC*_{camera}_*' + '.npy'))[0]
                    video_pth = os.path.join(video_dir, f'{video_name_list[i]}.{ext[i]}')
                    vidcap_list.append(cv2.VideoCapture(video_pth))
                    bbox_pth_list.append(bbox_pth)
                    mask_dir_list.append(glob(osp.join(mask_base_dir, f'*_{camera}*'))[0])
                save_pth_list = [osp.join(save_dir, subject, time, activity, camera + '.npy') for camera in camera_list]
                vis_pth_list = [osp.join(vis_dir, subject, time, activity, camera + '.mp4') for camera in camera_list]

                if all([osp.exists(save_pth) for save_pth in save_pth_list]):
                    print(f'{subject} {activity} already processed')
                    continue

                print(f'Running {subject} {time} {activity}  |  Save at (example) {save_pth_list[0]} ...')
                inference(cfg, vidcap_list, bbox_pth_list, mask_dir_list, Ks, model, save_pth_list, vis_pth_list, device, visualize=True)
                import pdb; pdb.set_trace()

if __name__ == '__main__':
    main()