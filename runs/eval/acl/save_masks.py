import os
import os.path as osp
import glob
import numpy as np

mask_base_dir = '/home/cmumbl/Data/ACL/samurai_results/masks'
subject_dir = glob.glob(osp.join(mask_base_dir, 'NIMBLE*'))

for subject in subject_dir:
    subject_name = osp.basename(subject)
    time_dirs = glob.glob(osp.join(subject, '*_months'))
    for time_dir in time_dirs:
        time_name = osp.basename(time_dir)
        activity_dirs = glob.glob(osp.join(time_dir, '*'))
        for activity_dir in activity_dirs:
            activity_name = osp.basename(activity_dir)
            mask_files = glob.glob(osp.join(activity_dir, '*.npz'))
            for mask_pth in mask_files:
                npz = np.load(mask_pth)
                key = npz.files[0]  
                mask_seq = npz[key]    # ndarray，shape = (T, H, W[, C])
                save_dir = mask_pth[:-4].replace('/masks/', '/masks_npy/')
                os.makedirs(save_dir, exist_ok=True)
                
                for i in range(mask_seq.shape[0]):
                    out_pth = osp.join(save_dir, f"{i:05d}.npy")
                    np.save(out_pth, mask_seq[i])
                
                print(f"{osp.basename(mask_pth)} → {mask_seq.shape[0]} frame, save to {save_dir}/")