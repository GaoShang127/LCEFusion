# coding:utf-8

import random
import torchvision.transforms.functional as TF
from PIL import Image
from natsort import natsorted
from torch.utils.data.dataset import Dataset
from utils import *
from pathlib import Path


class EnhanceDataset(Dataset):
    def __init__(self, split, dataset_name=None, lowimg_dir=None, img_color='Color'):
        super(EnhanceDataset, self).__init__()
        self.img_color = img_color
        assert split in ['train', 'test'], 'split must be "train"|"test"'
        # When the dataset type is set to 'train'
        if split == 'train':
            # Semi-open path
            self.lowimg_dir = os.path.join("./dataset/enhance", dataset_name)
            self.filelist = natsorted(os.listdir(self.lowimg_dir))
            self.split = split
            self.length = len(self.filelist)
        # When the dataset type is set to 'test'
        elif split == 'test':
            # Fully open path
            self.lowimg_dir = lowimg_dir
            self.filelist = natsorted(os.listdir(self.lowimg_dir))
            self.split = split
            self.length = len(self.filelist)

    def __getitem__(self, index):
        lowimg_name = self.filelist[index]
        lowimg_path = os.path.join(self.lowimg_dir, lowimg_name)
        lowimg = self.imread(lowimg_path, img_color=self.img_color)
        return lowimg, lowimg_name

    def __len__(self):
        return self.length

    @staticmethod
    def imread(path, img_color='Color'):
        # Read a color or grayscale image
        if img_color == 'Color':
            img = Image.open(path).convert('RGB')
        else:
            img = Image.open(path).convert('L')
        im_ts = TF.to_tensor(img)
        return im_ts


class AdjustmentDataset(Dataset):
    def __init__(self, split, dataset_name=None, img_dir=None, img_color=True, is_train=True):
        super(AdjustmentDataset, self).__init__()
        self.img_color = img_color
        self.is_train = is_train
        assert split in ['train', 'test'], 'split must be "train" or "test"'
        # When the dataset type is set to 'train'
        if split == 'train':
            self.img_dir = os.path.join("./dataset/correct", dataset_name)
        # When the dataset type is set to 'test'
        else:
            self.img_dir = img_dir
        valid_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        self.filelist = [f for f in natsorted(os.listdir(self.img_dir)) if f.lower().endswith(valid_ext)]
        self.length = len(self.filelist)

    def __getitem__(self, index):
        img_name = self.filelist[index]
        img_path = os.path.join(self.img_dir, img_name)
        img_gt = self.imread(img_path, img_color=self.img_color)
        # Generate a random color cast
        if self.is_train:
            p = random.random()
            # 1.Generate a normal sample with a 20% probability
            if p < 0.20:
                img_1 = img_gt.clone()
                img_2 = img_gt.clone()
            # 2.Generate a sample with a mild color cast with a 35% probability
            elif p < 0.55:
                img_1 = self.apply_mild_bias(img_gt)
                img_2 = self.apply_mild_bias(img_gt)
            # 3.Generate a sample with a complex color cast with a 35% probability.
            elif p < 0.90:
                if random.random() < 0.7:
                    # In most cases, both branches use the same color-shift type
                    mode = random.choice(['global', 'gradient'])
                    img_1 = self.apply_bias_by_mode(img_gt, mode)
                    img_2 = self.apply_bias_by_mode(img_gt, mode)
                else:
                    # In a few cases, the two branches use different color-shift types
                    mode_1 = random.choice(['global', 'gradient'])
                    mode_2 = random.choice(['global', 'gradient'])
                    img_1 = self.apply_bias_by_mode(img_gt, mode_1)
                    img_2 = self.apply_bias_by_mode(img_gt, mode_2)
            # 4. Generate a sample with mixed color shifts with a 10% probability
            else:
                img_1 = self.apply_mixed_bias(img_gt)
                img_2 = self.apply_mixed_bias(img_gt)
            return img_1, img_2, img_gt, img_name
        else:
            return img_gt, img_name

    def __len__(self):
        return self.length

    @staticmethod
    def apply_bias_by_mode(img, mode):
        if mode == 'global':
            return AdjustmentDataset.apply_global_bias(img)
        elif mode == 'gradient':
            return AdjustmentDataset.apply_gradient_bias(img)
        elif mode == 'mixed':
            return AdjustmentDataset.apply_mixed_bias(img)
        elif mode == 'mild':
            return AdjustmentDataset.apply_mild_bias(img)
        else:
            raise ValueError(f'Unsupported mode: {mode}')

    @staticmethod
    def apply_mild_bias(img):
        C, H, W = img.shape
        device = img.device
        # Slight channel gain
        gain = torch.empty((3, 1, 1), device=device).uniform_(0.94, 1.06)
        out = img * gain
        # Slight global bias
        if random.random() < 0.5:
            bias = torch.empty((3, 1, 1), device=device).uniform_(-0.015, 0.015)
            out = out + bias
        # Slight gamma perturbation
        if random.random() < 0.3:
            gamma = torch.empty((3, 1, 1), device=device).uniform_(0.98, 1.02)
            out = torch.clamp(out, 1e-4, 1.0) ** gamma
        return torch.clamp(out, 0.0, 1.0)

    @staticmethod
    def apply_global_bias(img):
        C, H, W = img.shape
        device = img.device
        # Moderate channel gain
        global_gain = torch.empty((3, 1, 1), device=device).uniform_(0.88, 1.12)
        out = img * global_gain
        # Moderate global bias
        if random.random() < 0.6:
            global_bias = torch.empty((3, 1, 1), device=device).uniform_(-0.025, 0.025)
            out = out + global_bias
        # Moderate gamma perturbation
        if random.random() < 0.4:
            gamma = torch.empty((3, 1, 1), device=device).uniform_(0.96, 1.04)
            out = torch.clamp(out, 1e-4, 1.0) ** gamma
        return torch.clamp(out, 0.0, 1.0)

    @staticmethod
    def apply_gradient_bias(img):
        C, H, W = img.shape
        device = img.device
        grid_size = random.choice([2, 3, 4])
        # Local gradient color shift
        low_res_gains = torch.empty((1, 3, grid_size, grid_size), device=device).uniform_(0.90, 1.10)
        spatial_gains = F.interpolate(low_res_gains, size=(H, W), mode='bicubic', align_corners=False).squeeze(0)
        spatial_gains = torch.clamp(spatial_gains, 0.88, 1.12)
        out = img * spatial_gains
        # Slight gamma perturbation
        if random.random() < 0.35:
            gamma = torch.empty((3, 1, 1), device=device).uniform_(0.98, 1.02)
            out = torch.clamp(out, 1e-4, 1.0) ** gamma
        return torch.clamp(out, 0.0, 1.0)

    @staticmethod
    def apply_mixed_bias(img):
        out = AdjustmentDataset.apply_weak_global_bias(img)
        out = AdjustmentDataset.apply_weak_gradient_bias(out)
        return torch.clamp(out, 0.0, 1.0)

    @staticmethod
    def apply_weak_global_bias(img):
        C, H, W = img.shape
        device = img.device
        gain = torch.empty((3, 1, 1), device=device).uniform_(0.92, 1.08)
        out = img * gain
        if random.random() < 0.5:
            bias = torch.empty((3, 1, 1), device=device).uniform_(-0.015, 0.015)
            out = out + bias
        if random.random() < 0.3:
            gamma = torch.empty((3, 1, 1), device=device).uniform_(0.98, 1.02)
            out = torch.clamp(out, 1e-4, 1.0) ** gamma
        return torch.clamp(out, 0.0, 1.0)

    @staticmethod
    def apply_weak_gradient_bias(img):
        C, H, W = img.shape
        device = img.device
        grid_size = random.choice([2, 3])
        low_res_gains = torch.empty((1, 3, grid_size, grid_size), device=device).uniform_(0.94, 1.06)
        spatial_gains = F.interpolate(low_res_gains, size=(H, W), mode='bicubic', align_corners=False).squeeze(0)
        spatial_gains = torch.clamp(spatial_gains, 0.92, 1.08)
        out = img * spatial_gains
        return torch.clamp(out, 0.0, 1.0)

    @staticmethod
    def imread(path, img_color=True):
        if img_color:
            img = Image.open(path).convert('RGB')
        else:
            img = Image.open(path).convert('L')
        img = img.resize((640, 480), Image.Resampling.BILINEAR)
        im_ts = TF.to_tensor(img)
        return im_ts


class FusionDataset(Dataset):
    def __init__(self, split, dataset_names=None, dataset_path=None):
        super(FusionDataset, self).__init__()
        assert split in ['train', 'val', 'test'], 'split must be "train"|"val"|"test"'
        # When the dataset type is set to 'train'
        if split == 'train':
            self.crop_size = 128
            self.dataset_nums = len(dataset_names)
            self.vis_paths_list = []
            self.ir_paths_list = []
            self.dataset_indexs = []
            self.img_names_list = []
            # Semi-open path
            for i in range( self.dataset_nums):
                vis_dir = os.path.join("./dataset/fusion", dataset_names[i], split, "vi")
                ir_dir = os.path.join("./dataset/fusion", dataset_names[i], split, "ir")
                vis_files, vis_paths = self.find_file(vis_dir)
                ir_files, ir_paths = self.find_file(ir_dir)
                assert len(vis_paths) == len(ir_paths)
                self.vis_paths_list.extend(vis_paths)
                self.ir_paths_list.extend(ir_paths)
                for vis_file in vis_files:
                    self.img_names_list.append(os.path.splitext(vis_file)[0])
                self.dataset_indexs.extend([i] * len(ir_paths))
                assert len(self.vis_paths_list) == len(self.ir_paths_list) == len(self.dataset_indexs)
            self.split = split
            self.length = len(self.vis_paths_list)

        # When the dataset type is set to 'test'
        elif split == 'val' or split == 'test':
            # Fully open path
            self.vis_dir = os.path.join(dataset_path, "vi")
            self.ir_dir = os.path.join(dataset_path, "ir")
            self.filelist = natsorted(os.listdir(self.vis_dir))
            self.split = split
            self.length = min(len(self.filelist), len(self.filelist))

    def __getitem__(self, index):
        if self.split == 'train':
            vis_path = self.vis_paths_list[index]
            ir_path = self.ir_paths_list[index]
            img_name = self.img_names_list[index]
        else:
            img_name = self.filelist[index]
            vis_path = os.path.join(self.vis_dir, img_name)
            ir_path = os.path.join(self.ir_dir, img_name)
        img_vis, img_ir = self.imread(vis_path, ir_path, self.split)
        return img_vis, img_ir, img_name

    def __len__(self):
        return self.length

    def find_file(self, dir):
        path = os.listdir(dir)
        if os.path.isdir(os.path.join(dir, path[0])):
            paths = []
            for dir_name in os.listdir(dir):
                for file_name in os.listdir(os.path.join(dir, dir_name)):
                    paths.append(os.path.join(dir, file_name, file_name))
        else:
            paths = list(Path(dir).glob('*'))
        return path, paths

    def imread(self, vis_path, ir_path, mode='train'):
        img_vis = Image.open(vis_path).convert('RGB')
        img_ir = Image.open(ir_path).convert('L')
        if mode == 'train':
            W, H = img_vis.size
            # Generate the initial coordinates for random cropping
            rnd_h = random.randint(0, max(0, H - self.crop_size))
            rnd_w = random.randint(0, max(0, W - self.crop_size))
            # Randomly crop the source image
            patch_vis = img_vis.crop((rnd_w, rnd_h, rnd_w + self.crop_size, rnd_h + self.crop_size))
            patch_ir = img_ir.crop((rnd_w, rnd_h, rnd_w + self.crop_size, rnd_h + self.crop_size))
            # Randomly generate a data augmentation mode (0-7) for the eight combinations
            patch_vis_np = np.array(patch_vis)
            patch_ir_np = np.array(patch_ir)
            augment_mode = random.randint(0,7)
            img_vis = augment_img(patch_vis_np, mode=augment_mode)
            img_ir = augment_img(patch_ir_np, mode=augment_mode)
            img_vis = np.clip(img_vis, 0, 255).astype('uint8')
            img_ir = np.clip(img_ir, 0, 255).astype('uint8')
            img_vis = Image.fromarray(img_vis)
            img_ir = Image.fromarray(img_ir)
        else:
            img_vis = img_vis
            img_ir = img_ir
        img_vis_tensor = TF.to_tensor(img_vis)
        img_ir_tensor = TF.to_tensor(img_ir)
        return img_vis_tensor, img_ir_tensor




