from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import os.path as osp
from collections import defaultdict

import torch
import joblib
import numpy as np
from tqdm import tqdm
from smplx import SMPLX

from configs import constants as _C

def process_amass():
    target_fps = 30
    
    _, datasets, _ = next(os.walk(_C.PATHS.AMASS_BASE_DIR))
    body_models = {
        gender: SMPLX(os.path.join(_C.PATHS.BODY_MODEL_DIR, 'smplx'), 
                      num_betas=16, 
                      ext='npz', 
                      flat_hand_mean=True, 
                      use_pca=False,
                      gender=gender).eval()
        for gender in ['male', 'female', 'neutral']
    }
    outdict = dict(
        poses=[],
        betas=[],
        trans=[],
        gender=[],
        sequence_idx=[]
    )
    
    sequence_idx = 0
    for dataset in (dataset_bar := tqdm(datasets, dynamic_ncols=True)):
        dataset_bar.set_description(f"Processing {dataset}")
        dataset_path = os.path.join(_C.PATHS.AMASS_BASE_DIR, dataset)

        seqs = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
        for seq in (seq_bar := tqdm(seqs, dynamic_ncols=True, leave=False)):
            seq_bar.set_description(f"Processing {seq}")
            seq_path = os.path.join(dataset_path, seq)
            for file in (file_bar := tqdm(os.listdir(seq_path), dynamic_ncols=True, leave=False)):
                file_bar.set_description(f"Processing {file}")
                if file.endswith('shape.npz') or file.endswith('stagei.npz'): 
                    # Skip shape and stagei files
                    continue
                
                if not file.endswith('.npz'):
                    continue

                data = dict(np.load(os.path.join(seq_path, file), allow_pickle=True))
                framerate_key = [k for k in data.keys() if 'mocap_frame' in k][0]
                mocap_framerate = data[framerate_key]
                retain_freq = int(mocap_framerate / target_fps + 0.5)
                F = len(data['poses'][::retain_freq])

                # Skip if the sequence is too short
                if F < 25: continue
                
                pose = data['poses'][::retain_freq].reshape(F, 165)
                betas = data['betas'][None].repeat(F, axis=0)
                trans = data['trans'][::retain_freq].reshape(F, 3)
                gender = data['gender'].item()
                # with torch.no_grad():
                #     output = body_models[gender](
                #         global_orient=torch.from_numpy(pose)[:, :3].float(),
                #         body_pose=torch.from_numpy(pose)[:, 3:66].float(),
                #         betas=torch.from_numpy(betas[:]).float(),
                #         transl=torch.from_numpy(trans).float(),
                #         left_hand_pose=torch.from_numpy(pose[:, 75:120]).float(),
                #         right_hand_pose=torch.from_numpy(pose[:, 120:165]).float(),
                #         jaw_pose=torch.from_numpy(pose[:, 66:69]).float(),
                #         leye_pose=torch.from_numpy(pose[:, 69:72]).float(),
                #         reye_pose=torch.from_numpy(pose[:, 72:75]).float(),
                #         expression=torch.zeros((F, 10)).float()
                #     )
                
                outdict['poses'].append(pose.astype(np.float32))
                outdict['betas'].append(betas.astype(np.float32))
                outdict['trans'].append(trans.astype(np.float32))
                outdict['gender'].append(np.array([gender] * F))
                outdict['sequence_idx'].append(np.array([sequence_idx] * F, dtype=np.int32))
                sequence_idx += 1
                
    for k in outdict.keys():
        outdict[k] = np.concatenate(outdict[k], axis=0)
        
    joblib.dump(outdict, _C.PATHS.AMASS_PARSED_LABEL_PTH)

if __name__ == '__main__':
    out_path = _C.PATHS.AMASS_PARSED_LABEL_PTH
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    process_amass()