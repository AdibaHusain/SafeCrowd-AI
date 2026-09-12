import os
import glob
import h5py
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class CrowdCountingDataset(Dataset):
    """
    Expects a folder structure like:
        data_dir/
            images/   *.jpg
            ground_truth/  *.h5   (same filename as image, precomputed density map)

    If your Colab pipeline produced ground truth in a different format
    (e.g. .mat files with point annotations), this is the ONE file to change —
    everything else (model, train loop) stays the same regardless of how
    ground truth is stored.
    """

    def __init__(self, data_dir: str):
        self.image_paths = sorted(glob.glob(os.path.join(data_dir, "images", "*.jpg")))
        self.gt_dir = os.path.join(data_dir, "ground_truth")

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in {data_dir}/images — check the path in config.py")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")

        gt_filename = os.path.splitext(os.path.basename(img_path))[0] + ".h5"
        gt_path = os.path.join(self.gt_dir, gt_filename)

        with h5py.File(gt_path, "r") as f:
            density_map = np.asarray(f["density"], dtype=np.float32)

        img_tensor = TRANSFORM(img)

        # CSRNet's backend has stride 8 relative to input due to the 3 max-pools
        # in the frontend, so the density map target must be downsampled to match
        # the model's output resolution, and rescaled to preserve total count.
        target = torch.from_numpy(density_map).unsqueeze(0).unsqueeze(0)
        target = torch.nn.functional.interpolate(
            target, scale_factor=0.125, mode="bilinear", align_corners=False
        ).squeeze(0) * 64  # 1/0.125^2 = 64, keeps sum(target) ≈ true head count

        return img_tensor, target