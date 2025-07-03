import os
import os.path as osp
import numpy as np
import cv2
from glob import glob

def get_video_list(video_dir):
    # full_camera_list = ["3175802", "3175947", "3175948", "3176142", "3176832", "3177649"]
    rotated_camera = "3177649"
    video_name_list = sorted(os.listdir(video_dir))
    
    if len(video_name_list) > 6:
        candidates = [video for video in video_name_list if rotated_camera in video]
        for candidate in candidates:
            if not candidate.startswith("Rotated"):
                video_name_list.remove(candidate)
    
    camera_list = []
    for video in video_name_list:
        if video.startswith("Rotated"):
            camera = video.split('_')[2]
        else:
            camera = video.split('_')[1]
        if camera not in camera_list:
            camera_list.append(camera)
    camera_list = sorted([camera for camera in camera_list])
    _video_name_list = [video.split('.')[0] for video in video_name_list]
    ext = [video.split('.')[-1] for video in video_name_list]

    return _video_name_list, ext, camera_list


def load_calibration(path, camera_list):
    trg_serials = camera_list
    
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