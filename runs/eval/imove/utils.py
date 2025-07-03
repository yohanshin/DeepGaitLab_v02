import os
import os.path as osp
from glob import glob
import math
import pickle
import torch
import numpy as np
from configs import constants as _C
import cv2
from tqdm import trange



def get_paths(exp_path):
    config_path = osp.join(exp_path, 'config.yaml')
    if 'ckpt' in os.listdir(exp_path):
        ckpt = glob(osp.join(exp_path, 'ckpt', '*.pth'))
    else:
        ckpt = glob(osp.join(exp_path, '*.pth'))
    epochs = [int(osp.basename(c).split('_')[1]) for c in ckpt]
    
    # Get the latest checkpoint
    ckpt = [c for _, c in sorted(zip(epochs, ckpt))]
    ckpt = ckpt[-1]
    return config_path, ckpt


def get_video_list(video_dir):
    full_camera_list = ["C11398", "C11399", "C11400", "C11409", "C11410", "C11411", "C11412", "C11413", "C11414", "C11415"]
    rotated_cameras = ["Camera_22", "Camera_23"]
    video_name_list = sorted(os.listdir(video_dir))
    video_name_list = sorted([video for video in video_name_list if not video.startswith('.')])
    
    if len(video_name_list) > 10:
        candidates = [video for video in video_name_list if "Camera_22" in video or "Camera_23" in video]
        for candidate in candidates:
            cap = cv2.VideoCapture(os.path.join(video_dir, candidate))
            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            
            # # Remove not rotated video
            # if width > 1500:
            #     video_name_list.remove(candidate)
    if len(video_name_list) > 10:
        candidates = [video for video in video_name_list if "C11399" in video or "C11400" in video]
        for candidate in candidates:
            if not candidate.startswith("Rotated"):
                video_name_list.remove(candidate)
        # video_name_list = sorted([video for video in video_name_list if not video.startswith("Rotated")])

    camera_list = [video.split('(')[1].split(')')[0] for video in video_name_list]
    camera_idx_list = sorted([i for i, camera in enumerate(full_camera_list) if camera in camera_list])
    camera_list = sorted([camera for camera in camera_list])
    _video_name_list = []
    for camera in camera_list:
        for video_name in video_name_list:
            if camera in video_name:
                _video_name_list.append(video_name)

    return _video_name_list, camera_idx_list, camera_list


def load_calibration(path, camera_list):
    trg_serials = [str(21 + camera) for camera in camera_list]
    
    import xml.etree.ElementTree as ET
    
    tree = ET.parse(path)
    root = tree.getroot()
    
    Ks, Rs, Ts, dists, serials = [], [], [], [], []
    cameras = []
    # loop over all cameras
    for camera in root.findall('.//camera'):
        serial = camera.get('serial')
        if not serial in trg_serials: continue

        serials.append(camera.get('serial'))
        
        K = np.eye(3)
        R = np.eye(3)
        T = np.zeros(3)
        dist = np.zeros(5)
    
        # Get camera extrinsics
        transform = camera.find('transform')
        T[0] = float(transform.get('x'))
        T[1] = float(transform.get('y'))
        T[2] = float(transform.get('z'))
        R[0, 0] = float(transform.get('r11'))
        R[0, 1] = float(transform.get('r12'))
        R[0, 2] = float(transform.get('r13'))
        R[1, 0] = float(transform.get('r21'))
        R[1, 1] = float(transform.get('r22'))
        R[1, 2] = float(transform.get('r23'))
        R[2, 0] = float(transform.get('r31'))
        R[2, 1] = float(transform.get('r32'))
        R[2, 2] = float(transform.get('r33'))
        
        # get the camera intrinsic parameters
        intrinsic = camera.find('intrinsic')
        K[0, 0] = float(intrinsic.get('focalLengthU'))
        K[1, 1] = float(intrinsic.get('focalLengthV'))
        K[0, 2] = float(intrinsic.get('centerPointU'))
        K[1, 2] = float(intrinsic.get('centerPointV'))
        dist[0] = float(intrinsic.get('radialDistortion1'))
        dist[1] = float(intrinsic.get('radialDistortion2'))
        dist[2] = float(intrinsic.get('tangentalDistortion1'))
        dist[3] = float(intrinsic.get('tangentalDistortion2'))
        dist[4] = float(intrinsic.get('radialDistortion3'))
        
        Ks.append(K)
        Rs.append(R)
        Ts.append(T / 1e3)
        dists.append(dist)
        cameras.append(camera.get('serial'))

    params = {}
    for i, camera in enumerate(cameras):
        params[camera] = {}
        params[camera]['K'] = Ks[i]
        params[camera]['R'] = Rs[i]
        params[camera]['T'] = Ts[i]
        params[camera]['dist'] = dists[i]

    return np.stack(Ks), np.stack(Rs), np.stack(Ts), np.stack(dists), serials


def load_data(detector, subject, activity, camera_list, camera_idx_list, downsample_kpts=False, downsample_bbox=False):
    kpts, bboxes = [], []
    for camera in camera_list:
        
        # Load 2D keypoints
        kpt_fname = _C.PATHS.D10.KPT_DETECTION_PTH.replace('detector', detector)
        
        for after, before in zip([subject, activity, camera], ['subject', 'activity', 'camera']):
            kpt_fname = kpt_fname.replace(before, after)
        kpt_fname = kpt_fname.replace(camera+"s", "cameras")
        kpt_fname = glob(kpt_fname)
        
        if not len(kpt_fname) == 1:
            import pdb; pdb.set_trace()
        assert len(kpt_fname) == 1
        kpt_fname = kpt_fname[0]
        kpts.append(np.load(kpt_fname))
        
        # Load bbox
        bbox_fname = _C.PATHS.D10.BBOX_PTH
        for after, before in zip([subject, activity, camera], ['subject', 'activity', 'camera']):
            bbox_fname = bbox_fname.replace(before, after)
        bbox_fname = glob(bbox_fname)
        # if not len(bbox_fname) == 1:
        #     import pdb; pdb.set_trace()
        assert len(bbox_fname) == 1
        bbox_fname = bbox_fname[0]
        bboxes.append(np.load(bbox_fname))

    kpts = np.stack(kpts, axis=1)
    if 'wholebody' in detector.lower():
        kpts = kpts[..., :23, :]    # Only use 23 body+feet keypoints

    bboxes = np.stack(bboxes, axis=1)

    Ks, Rs, Ts, dists, serials = load_calibration(
        _C.PATHS.D10.CALIB_PTH.replace('subject', subject), camera_idx_list
    )
    calibs = {
        'Ks': Ks,
        'Rs': Rs,
        'Ts': Ts,
        'dists': dists,
        'serials': serials
    }

    if downsample_kpts:
        kpts = kpts[::3]
    if downsample_bbox:
        bboxes = bboxes[::3]

    if bboxes.shape[0] > kpts.shape[0]:
        bboxes = bboxes[:kpts.shape[0]]
    elif bboxes.shape[0] < kpts.shape[0]:
        import pdb; pdb.set_trace()
    
    return kpts, bboxes, calibs


def save(data, filepath):
    with open(filepath, 'wb') as file:
        pickle.dump(data, file)


def load(filepath):
    with open(filepath, 'rb') as file:
        data = pickle.load(file)
    return data