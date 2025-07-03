import os
import sys
sys.path.append('./')
import glob
import json
import argparse

import cv2
import shlex
import torch
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from llib.vis.vis_fn import visualize_bbox
from runs.demo.subprocess import run_command_with_conda

YOLO_CKPT = "/fast/sshin/checkpoints/yolov8x.pt"
DEEPGAITLAB_MODEL_EXP_DIR = "/fast/sshin/projects/DeepGaitLab_v02/experiments/train/bedlamlab_2d/pose_detr-ViT-Large-tune-mask-surface"
SAMURAI_WORKING_DIR = "/fast/sshin/projects/srcs/samurai"
DEEPGAITLAB_WORKING_DIR = "/fast/sshin/projects/DeepGaitLab_v02"
SAMURAI_CONDA_ENV = "samurai"
DEEPGAITLAB_CONDA_ENV = "dgl_env"

def get_initial_prompt(image, tmp_fldr):
    # Get the bounding box of the person in the image
    model = YOLO(YOLO_CKPT)
    bboxes = model.predict(
        image, device=device, classes=0, conf=0.5, save=False, verbose=False
    )[0].boxes
    
    _bboxes = []
    vis_image = image.copy()
    for i, bbox in enumerate(bboxes):
        xyxy = bbox[0].xyxy.detach().cpu().numpy()
        cxywh = bbox[0].xywh.detach().cpu().numpy()
        _bbox = np.zeros_like(xyxy)
        _bbox[..., :2] = xyxy[..., :2]
        _bbox[..., 2:] = cxywh[..., 2:]
        _bboxes.append(_bbox)
        if i == 0: 
            color = (0, 255, 0)
        else:
            color = (255, 0, 0)

        vis_image = visualize_bbox(_bbox, vis_image.copy(), color, bbox_id=i + 1)
    cv2.imwrite(os.path.join(tmp_fldr, 'init_prompt.png'), vis_image)
    print(f"Saved initial prompt to {os.path.join(tmp_fldr, 'init_prompt.png')}")
    user_input = input("Target bbox ID: ")
    bbox_id = int(user_input)
    target_bbox = _bboxes[bbox_id - 1]
    os.system(f"rm -rf {os.path.join(tmp_fldr, 'init_prompt.png')}")
    np.save(os.path.join(tmp_fldr, 'init_prompt.npy'), target_bbox)

    return os.path.join(tmp_fldr, 'init_prompt.npy')


def run_samurai(video_path, output_fldr, prompt, tmp_fldr):
    cmd = ["python", "-u",            # -u = unbuffered stdout/stderr
            "run_on_video.py", 
            "--input_video", shlex.quote(video_path), 
            "--prompt", shlex.quote(prompt), 
            "--output_fldr", shlex.quote(os.path.abspath(output_fldr)), 
            "--tmp_fldr", shlex.quote(tmp_fldr), 
            ]
    
    run_command_with_conda(SAMURAI_WORKING_DIR, SAMURAI_CONDA_ENV, cmd)


def run_dgl(video_path, output_fldr, bbox_pth, mask_dir):
    cmd = ["python", "-u",            # -u = unbuffered stdout/stderr
           "-m" "runs.demo.base_run", 
           "--config-path", DEEPGAITLAB_MODEL_EXP_DIR, 
           f'demo.input_video={video_path}', 
           f'demo.output_dir={output_fldr}', 
           f'demo.bbox_pth={bbox_pth}', 
           f'demo.mask_dir={mask_dir}', 
           'demo.visualize=True', 
           ]
    run_command_with_conda(DEEPGAITLAB_WORKING_DIR, DEEPGAITLAB_CONDA_ENV, cmd)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_video', default='')
    parser.add_argument('-o', '--output_fldr', default='outputs/demo/video')
    parser.add_argument('-t', '--tmp_fldr', default='~/tmp/dgl_demo')
    parser.add_argument('-m', '--model', default='')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Change the name of the video
    input_video = args.input_video
    import pdb; pdb.set_trace()
    # input_video = args.input_video.replace(' ', '\ ').replace('(', '\(').replace(')', '\)')
    # os.system(f"mv {args.input_video} {input_video}")
    
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file {input_video}")
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create output folder if it doesn't exist
    output_fldr = os.path.join(args.output_fldr, os.path.basename(input_video).split('.')[0])
    output_fldr = output_fldr.replace(' ', '_')
    os.makedirs(output_fldr, exist_ok=True)

    bbox_pth = os.path.join(output_fldr, 'samurai', 'bbox.npy')
    if not os.path.exists(bbox_pth):
        # Create temporary folder if it doesn't exist
        tmp_fldr = os.path.expanduser(args.tmp_fldr)
        os.makedirs(tmp_fldr, exist_ok=True)

        # Initialize YOLO model
        ret, init_frame = cap.read()
        cap.release()
        
        prompt = get_initial_prompt(init_frame, tmp_fldr)
        run_samurai(args.input_video, output_fldr, prompt, tmp_fldr)
        os.system(f"rm -rf {tmp_fldr}")
    else:
        print(f"Bbox already exists at {bbox_pth}")

    bbox_pth = os.path.join(output_fldr, 'samurai', 'bbox.npy')
    mask_dir = os.path.join(output_fldr, 'samurai', 'masks')
    run_dgl(args.input_video, output_fldr, bbox_pth, mask_dir)