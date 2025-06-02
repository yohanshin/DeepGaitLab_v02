import numpy as np

marker_names = [
    'APEX', 'LASI', 'RASI', 'LPS2', 'RPS2', 
    'LHJC', 'RHJC', 'LFLT', 'LFFT', 'LFBT', 
    'LFLB', 'LFBB', 'LFMB', 'LFFB', 'LLEP', 
    'LMEP', 'RFLT', 'RFFT', 'RFBT', 'RFLB', 
    'RFBB', 'RFMB', 'RFFB', 'RLEP', 'RMEP', 
    'LTTB', 'LTIB', 'FLTI', 'BLTI', 'LTIC', 
    'LLML', 'LMML', 'RTTB', 'RTIB', 'FRTI', 
    'BRTI', 'RTIC', 'RLML', 'RMML', 'LCAL', 
    'L1MT', 'L2MT', 'L5MT', 'RCAL', 'R1MT', 
    'R2MT', 'R5MT', 'C7_study', 'r_shoulder_study', 'L_shoulder_study', 
    'r_lelbow_study', 'r_melbow_study', 'r_lwrist_study', 'r_mwrist_study', 'L_lelbow_study', 
    'L_melbow_study', 'L_lwrist_study', 'L_mwrist_study'
]

l_hand_idx = sorted([marker_names.index(i) for i in ['L_lwrist_study', 'L_mwrist_study']])
r_hand_idx = sorted([marker_names.index(i) for i in ['r_lwrist_study', 'r_mwrist_study']])
l_feet_idx = sorted([marker_names.index(i) for i in ['LMML', 'LLML', 'LCAL', 'L1MT', 'L2MT', 'L5MT']])
r_feet_idx = sorted([marker_names.index(i) for i in ['RMML', 'RLML', 'RCAL', 'R1MT', 'R2MT', 'R5MT']])
head_idx = sorted([marker_names.index(i) for i in ['APEX', 'C7_study']])
body_idx = sorted([i for i in range(len(marker_names)) if not i in l_hand_idx + r_hand_idx + l_feet_idx + r_feet_idx + head_idx])

body_parts_dict = {
    "body": body_idx,
    "left_hand": l_hand_idx,
    "right_hand": r_hand_idx,
    "left_feet": l_feet_idx,
    "right_feet": r_feet_idx,
    "head": head_idx,
}
body_idx = np.array(body_idx + l_hand_idx + r_hand_idx + l_feet_idx + r_feet_idx + head_idx)

left_names = [m for m in marker_names if m.startswith("L")] + ["BLTI", "FLTI"]
right_names = [m for m in marker_names if m.startswith("R")] + [m for m in marker_names if m.startswith("r_")] + ["BRTI", "FRTI"]
left_idx = [marker_names.index(m) for m in left_names]
right_idx =[marker_names.index(m) for m in right_names]
flip_pairs = [[left_i, right_i] for left_i, right_i in zip(left_idx, right_idx)]

subsample_pts_fn = "/is/cluster/fast/sshin/data/body_models/osim_related/wo_upper_body/smplx_to_dgl.pkl"