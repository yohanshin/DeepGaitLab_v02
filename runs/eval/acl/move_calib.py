import os
import os.path as osp
import glob
import shutil


dir = '/media/cmumbl/T9/ACL_PT_videos/'
dst_dir = '/home/cmumbl/Data/ACL/calib/'
subject_dirs = glob.glob(osp.join(dir, 'NIMBLE*'))
for subject_dir in subject_dirs:
    subject = osp.basename(subject_dir)
    time_dirs = glob.glob(osp.join(subject_dir, '*_months'))
    for time_dir in time_dirs:
        time = osp.basename(time_dir)
        calib_src = osp.join(time_dir, 'PTCTRC', f'{subject}_{time}_PTCTRC_calibration.txt')
        calib_dst = osp.join(dst_dir, subject, time, f'calibration.txt')
        if not osp.exists(osp.dirname(calib_dst)):
            os.makedirs(osp.dirname(calib_dst))
        try:
            shutil.copy2(calib_src, calib_dst)
            print(f'Copied {calib_src} → {calib_dst}')
        except Exception as e:
            print(f'[ERROR] failed to copy {calib_src}: {e}')