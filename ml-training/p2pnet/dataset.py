import os
import glob
import random
import numpy as np
import cv2
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset
from torchvision import transforms

NORM = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def _load_points(mat_path: str) -> np.ndarray:
    """
    Standard ShanghaiTech .mat structure:
        mat["image_info"][0, 0][0, 0][0] -> (N, 2) array of (x, y) head points.
    Same raw annotation CSRNet's density maps were generated from — P2PNet
    consumes the points directly instead.
    """
    mat = loadmat(mat_path)
    points = mat["image_info"][0, 0][0, 0][0]
    return points.astype(np.float32)


class P2PDataset(Dataset):
    def __init__(self, data_dir: str, train: bool, crop_size: int = 256, limit_samples=None):
        self.image_paths = sorted(glob.glob(os.path.join(data_dir, "images", "*.jpg")))
        if limit_samples:
            self.image_paths = self.image_paths[:limit_samples]
        self.gt_dir = os.path.join(data_dir, "ground_truth")
        self.train = train
        self.crop_size = crop_size

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in {data_dir}/images — check config.py paths")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        base = os.path.splitext(os.path.basename(img_path))[0]  # e.g. "IMG_1"
        gt_path = os.path.join(self.gt_dir, f"GT_{base}.mat")

        img = cv2.imread(img_path)
        if img is None:
            raise RuntimeError(f"cv2 failed to read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        points = _load_points(gt_path)

        if self.train:
            img, points = self._random_crop(img, points, self.crop_size)
            if random.random() > 0.5:
                w = img.shape[1]
                img = np.ascontiguousarray(img[:, ::-1, :])