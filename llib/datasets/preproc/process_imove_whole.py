import os
import sys
sys.path.append("./")
import os.path as osp
from glob import glob
from collections import defaultdict

import cv2
import torch
import joblib
import numpy as np
from tqdm import tqdm

torch.utils.data._utils.worker.IS_DAEMON = False
os.environ["PYTHONFAULTHANDLER"] = "1"
from loguru import logger

from runs.eval.inference import tools
from runs.eval.imove.utils import load_calibration, get_video_list

DATA_BASE_DIR = "/is/cluster/fast/sshin/data/D10_data"
CALIB_PTH = f'{DATA_BASE_DIR}/calibration/subject/calibration.txt'
RESULTS_DIR = f"{DATA_BASE_DIR}/DeepGaitLab_v02"

if __name__ == "__main__":
    image_size = (384, 512)
    with open("datasets/imove_val_list.txt", 'r', encoding='utf-8') as f:
        validation_txt_list = [line.strip() for line in f]

    # Data loading
    subject_list = sorted(os.listdir(osp.join(DATA_BASE_DIR, 'resampled_videos')))
    subject_list = [subject for subject in subject_list if not subject.startswith('.')]
    activity_list = sorted(os.listdir(osp.join(DATA_BASE_DIR, 'resampled_videos', 'Subject_15', 'RGB_outputs')))
    activity_list = [activity for activity in activity_list if not (activity.startswith('.'))]

    full_dataset = defaultdict(list)
    for subject in subject_list:
        # for activity in activity_list:
        for activity in ["t1_walking"]:
            video_pth = osp.join(DATA_BASE_DIR, 'resampled_videos', subject, 'RGB_outputs', activity)
            if not os.path.exists(video_pth):
                print(f'{subject} {activity} not found')
                continue

            # if "walking" in activity: continue

            # if os.path.exists(f"datasets/imove/imove_{subject}_{activity}.pkl"):
            #     print(f'{subject} {activity} already processed')
            #     continue

            seq_dataset = defaultdict(list)
            video_name_list, camera_idx_list, camera_list = get_video_list(video_pth)
            if len(video_name_list) != 10:
                continue

            video_dir = osp.join(DATA_BASE_DIR, 'resampled_videos', subject, 'RGB_outputs', activity)
            bbox_dir = osp.join(DATA_BASE_DIR, 'Refined_bbox', subject, activity)
            mask_base_dir = os.path.join(DATA_BASE_DIR, "DeepGaitLab_v02", "samurai", subject, activity)

            vidcap_list, bbox_pth_list, mask_dir_list = [], [], []
            for i, camera in enumerate(camera_list):
                if camera in ["C11399", "C11400"]: 
                    bbox_pth = glob(osp.join(bbox_dir, f'Rotated*{activity}*{camera}*' + '.npy'))[0]
                else:
                    bbox_pth = glob(osp.join(bbox_dir, f'{activity}*{camera}*' + '.npy'))[0]
                video_pth = os.path.join(video_dir, video_name_list[i])
                vidcap_list.append(cv2.VideoCapture(video_pth))
                bbox_pth_list.append(bbox_pth)
                mask_dir_list.append(os.path.join(mask_base_dir, camera))
            
            for cap, bbox_pth, mask_dir, camera in zip(vidcap_list, bbox_pth_list, mask_dir_list, camera_list):
                dataset = defaultdict(list)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                bbox = np.load(bbox_pth)
                if bbox.shape[0] > frame_count * 2:
                    bbox = bbox[::3]

                os.makedirs(os.path.join(DATA_BASE_DIR, "validation", subject, activity, camera, 'images',), exist_ok=True)
                os.makedirs(os.path.join(DATA_BASE_DIR, "validation", subject, activity, camera, 'masks',), exist_ok=True)
                
                frame_i = 0
                pbar = tqdm(total=frame_count, desc=f"Processing {subject} {activity} {camera}", leave=False, dynamic_ncols=True)
                while True:
                    ret, img = cap.read()
                    pbar.update(1)
                    if not ret:
                        break
                    
                    frame_i += 1
                    
                    if frame_i >= bbox.shape[0]:
                        break
                    
                    imagepth = os.path.join(DATA_BASE_DIR, "validation", subject, activity, camera, 'images', f"{frame_i:05}.jpg")
                    maskpth = os.path.join(DATA_BASE_DIR, "validation", subject, activity, camera, 'masks', f"{frame_i:05}.npy")
                    
                    if not os.path.exists(os.path.join(mask_dir, f"{frame_i:05}.npy")):
                        continue
                    
                    if np.all(bbox[frame_i] == 0.0):
                        dataset['imagepths'].append("")
                        dataset['maskpths'].append("")
                        dataset['center'].append(np.zeros(2))
                        dataset['scale'].append(np.zeros(2).max())
                        dataset['frame_ids'].append(frame_i)
                        dataset['cameras'].append(False)
                        continue
                    
                    center, scale = tools.xyxy2cs(bbox[frame_i], 0.75, pixel_std=200.0)
                    scale = scale.max()
                    
                    if not os.path.exists(imagepth):
                    # if True:
                        cvimg = tools.process_cvimg(img.copy(), center, scale, image_size, transform_only=True)
                        cv2.imwrite(imagepth, cvimg)
                    # if True:
                    if not os.path.exists(maskpth):
                        samurai_pth = os.path.join(mask_dir, f"{frame_i:05}.npy")
                        mask = np.load(samurai_pth).astype(np.float32) 
                        mask = tools.process_cvimg(mask.copy(), center, scale, image_size, transform_only=True)
                        np.save(maskpth, mask)

                    dataset['imagepths'].append(imagepth)
                    dataset['maskpths'].append(maskpth)
                    dataset['center'].append(center)
                    dataset['scale'].append(scale)
                    dataset['frame_ids'].append(frame_i)
                    dataset['cameras'].append(True)

                pbar.close()
                for key in dataset.keys():
                    seq_dataset[key].append(dataset[key])

            l = min([len(frame_ids) for frame_ids in seq_dataset['frame_ids']])
            for key, val in seq_dataset.items():
                seq_dataset[key] = np.stack([v[:l] for v in val], axis=1)

            out_dataset = dict()
            for key, val in seq_dataset.items():
                out_dataset[key] = val
            joblib.dump(out_dataset, f"datasets/imove/imove_{subject}_{activity}.pkl")