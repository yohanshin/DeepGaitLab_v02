import os

# Define all the required paths
class PATHS:
    class D10:
        if os.getcwd().split('/')[1] == 'ocean':
            BASE_DIR = '/ocean/projects/mch220002p/shared/D10_benchmark'
        elif 'sshin' in os.getcwd():
            BASE_DIR = '/fast/sshin/data/D10_data'
        else:
            assert 'soyongs' in os.getcwd()
            BASE_DIR = '/home/soyongs/Data/D10'
        
        # Metadata
        METADATA_PTH = f'{BASE_DIR}/metadata.json'
        
        # Video
        VIDEO_PTH = f'{BASE_DIR}/resampled_videos/subject/RGB_outputs/activity/*camera*.mp4'

        # Camera calibration
        CALIB_PTH = f'{BASE_DIR}/calibration/subject/calibration.txt'

        # 2D detection results paths
        BBOX_PTH = f'{BASE_DIR}/Refined_bbox/subject/activity/*camera*.npy'
        KPT_DETECTION_PTH = f'{BASE_DIR}/detection/detector/subject/activity/*camera*.npy'
        VITPOSE_PTH = f'{BASE_DIR}/detection/ViTPose/subject/activity/*camera*.npy'
        VITPOSE_WHOLEBODY_PTH = f'{BASE_DIR}/detection/ViTPose_wholebody/subject/activity/*camera*.npy'
        VITPOSE_BEDLAM_PTH = f'{BASE_DIR}/detection/detector/subject/activity/*camera*.npy'
        RANSAC_MASK_PTH = f'{BASE_DIR}/smplifyx_results/subject/activity/ransac_mask.pkl'

        # Reconstruction results paths
        SMPLIFYX_PKL_PTH = f'{BASE_DIR}/smplifyx_results/detector/subject/activity/results.pkl'
        SMPLIFYX_MP4_PTH = f'{BASE_DIR}/smplifyx_results/detector/subject/activity/results.mp4'
        SMPLIFYX_TRI_PTH = f'{BASE_DIR}/smplifyx_results/detector/subject/activity/triangulation.npy'

        # OpenSim results
        OPENSIM_XML_PTH = f'{BASE_DIR}/OpenSim_results/detector/subject/xmls'
        OPENSIM_MARKER_PTH = f'{BASE_DIR}/OpenSim_results/detector/subject/activity/marker.trc'
        OPENSIM_SCALE_PTH = f'{BASE_DIR}/OpenSim_results/detector/scaled_models/subject.osim'
        OPENSIM_MOTION_PTH = f'{BASE_DIR}/OpenSim_results/detector/subject/activity/ik.mot'

        # IK results
        IK_RESULTS_PTH = f'{BASE_DIR}/IK_results/subject/cmu_raj/activity_001/ik.mot'
        OPENCAP_RESULTS_PTH = f'{BASE_DIR}/OpenCap_2_cams/subject/OpenCap_2_cams_subject_results.pk'
        WHAM_RESULTS_PTH = f'{BASE_DIR}/WHAM_opt_cam/subject/WHAM_opt_cam_subject_results.pk'
        THEIA_RESULTS_PTH = f'{BASE_DIR}/Theia3D/subject/Theia3D_subject_results.pk'

        # Activity segmentation
        SEGMENTATION_PTH = f'{BASE_DIR}/data_segmentation/subject/activity/*.pkl'

    if os.getcwd().split('/')[1] == 'ocean':
        # SMPL / SMPL-X related
        BODY_MODEL_DIR = '/ocean/projects/mch220002p/shared/body_models'
        # Pretrained models
        VPOSER_PTH = f'{BODY_MODEL_DIR}/vposer_v02'

    elif 'sshin' in os.getcwd():
        # Pretrained models
        VPOSER_PTH = '/fast/sshin/data/body_models/vposer_v02'
        # SMPL / SMPL-X related
        BODY_MODEL_DIR = '/fast/sshin/data/body_models'

    else:
        assert 'soyongs' in os.getcwd()
        # Pretrained models
        VPOSER_PTH = '/home/soyongs/Data/body_models/vposer_v02'
        # SMPL / SMPL-X related
        BODY_MODEL_DIR = '/home/soyongs/Data/body_models'

    class OSIM:
        OSIM_DATA_DIR = '/home/soyongs/Data/body_models/osim_data'
        OSIM_MODEL_PTH = f'{OSIM_DATA_DIR}/model_fix_range.osim'
        SMPL_TO_MARKERS_REG_PTH = f'{OSIM_DATA_DIR}/smpl_to_rizzoli_29.pkl'
        MARKER_SETTING_PTH = f'{OSIM_DATA_DIR}/markers.xml'
        SCALE_SETTING_PTH = f'{OSIM_DATA_DIR}/scale_setting_default.xml'
        IK_SETTING_PTH = f'{OSIM_DATA_DIR}/subject_activity_ik.xml'

    SMPLX2SMPL = f'{BODY_MODEL_DIR}/smplx2smpl.pkl'
    DOWNSAMPLE_MAT = f'{BODY_MODEL_DIR}/downsample_mat_smplx.pkl'
    
    AMASS_BASE_DIR = "/fast/sshin/data/AMASS/SMPLX+G"
    AMASS_PARSED_LABEL_PTH = "datasets/parsed_data/amass_smplx.pth"