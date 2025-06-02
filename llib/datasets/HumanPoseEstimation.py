import numpy as np
from torchvision import transforms
# from torch.utils.data import Dataset

class DatasetRegistration(type):
    """
    Metaclass for registering different datasets
    """
    def __init__(cls, name, bases, nmspc):
        super().__init__(name, bases, nmspc)
        if not hasattr(cls, 'registry'):
            cls.registry = dict()
        cls.registry[name] = cls

    # Metamethods, called on class objects:
    def __iter__(cls):
        return iter(cls.registry)

    def __str__(cls):
        return str(cls.registry)

class Dataset(metaclass=DatasetRegistration):
    """
    Base Dataset class
    """
    def __init__(self, *args, **kwargs):
        pass

class HumanPoseEstimationDataset(Dataset):
    """
    HumanPoseEstimationDataset class.

    Generic class for HPE datasets.
    """
    def __init__(self, is_train=True, image_width=288, image_height=384,
                 scale=True, scale_factor=0.35, flip_prob=0.5, rotate_prob=0.5, trans_prob=0.5, rotation_factor=45., half_body_prob=0.3, trans_factor=0.0,
                 use_different_joints_weight=False, extreme_cropping_prob=0.1, img_aug_prob=0.9, heatmap_sigma=2, max_res=10, num_joints=512, total_num_joints=512,
                 flip_pairs=None, joints_weight=None,
                 **kwargs):

        self.max_res = max_res

        self.is_train = is_train
        self.scale = scale  # ToDo Check
        self.scale_factor = scale_factor
        self.trans_factor = trans_factor
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.trans_prob = trans_prob
        self.rotation_factor = rotation_factor
        self.half_body_prob = half_body_prob
        self.use_different_joints_weight = use_different_joints_weight  # ToDo Check
        self.extreme_cropping_prob = extreme_cropping_prob
        self.img_aug_prob = img_aug_prob
        self.heatmap_sigma = heatmap_sigma

        self.image_size = (image_width, image_height)
        self.aspect_ratio = image_width * 1.0 / image_height

        self.heatmap_size = (int(image_width / 4), int(image_height / 4))
        self.heatmap_type = 'gaussian'
        self.pixel_std = 200  # I don't understand the meaning of pixel_std (=200) in the original implementation

        self.num_joints = num_joints
        self.max_num_joints = total_num_joints
        self.num_joints_half_body = 8

        if flip_pairs:
            self.flip_pairs = flip_pairs
        if joints_weight:
            self.joints_weight = np.array(joints_weight).reshape(self.max_num_joints, 1)
        else:
            self.use_different_joints_weight = False

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Default values
        self.bbox_thre = 1.0
        self.image_thre = 0.0
        self.in_vis_thre = 0.2
        self.nms_thre = 1.0
        self.oks_thre = 0.9

        # Default values for target generation
        self.unbiased = False
        self.unbiased_encoding = True

    def _generate_target(self, joints, joints_vis, score_invis=0.5):
        """
        :param joints:  [num_joints, 2]
        :param joints_vis: [num_joints, 1]
        :return: target, target_weight(1: visible, score_invis: invisible)
        """
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        
        try: 
            target_weight[:, 0] = joints_vis[:, 0]
            target_weight[target_weight == 0] = score_invis
        except: 
            pass 

        nooi = 0
        if self.heatmap_type == 'gaussian':
            target = np.zeros((self.num_joints,
                               self.heatmap_size[1],
                               self.heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = self.heatmap_sigma * 3

            for joint_id in range(self.num_joints):
                feat_stride = np.asarray(self.image_size) / np.asarray(self.heatmap_size)
                
                if self.unbiased_encoding:
                    mu_x = joints[joint_id][0] / feat_stride[0]
                    mu_y = joints[joint_id][1] / feat_stride[1]
                    
                    ul = [mu_x - tmp_size, mu_y - tmp_size]
                    br = [mu_x + tmp_size + 1, mu_y + tmp_size + 1]

                    if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                            or br[0] < 0 or br[1] < 0:
                        # If not, just return the image as is
                        target_weight[joint_id] = 0
                        nooi += 1
                        continue
                    
                    size = 2 * tmp_size + 1
                    x = np.arange(0, self.heatmap_size[0], 1, np.float32)
                    y = np.arange(0, self.heatmap_size[1], 1, np.float32)
                    y = y[:, np.newaxis]

                    if target_weight[joint_id] > 0.5:
                        target[joint_id] = np.exp(-((x - mu_x)**2 +
                                                (y - mu_y)**2) /
                                                (2 * self.heatmap_sigma**2))
                        
                
                else:
                    mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
                    mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                    # Check that any part of the gaussian is in-bounds
                    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                    if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                            or br[0] < 0 or br[1] < 0:
                        # If not, just return the image as is
                        target_weight[joint_id] = 0
                        nooi += 1
                        continue

                    # # Generate gaussian
                    size = 2 * tmp_size + 1
                    x = np.arange(0, size, 1, np.float32)
                    y = x[:, np.newaxis]
                    x0 = y0 = size // 2
                    # The gaussian is not normalized, we want the center value to equal 1
                    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.heatmap_sigma ** 2))

                    # Usable gaussian range
                    g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                    g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                    # Image range
                    img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                    img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                    v = target_weight[joint_id]
                    if v >= 0.5:
                        target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                            g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        else:
            raise NotImplementedError

        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)

        return target, target_weight