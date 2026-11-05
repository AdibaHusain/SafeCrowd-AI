from config import CSRNetConfig
from dataset import CrowdCountingDataset

cfg = CSRNetConfig()

train_dataset = CrowdCountingDataset(cfg.train_data_dir)
test_dataset = CrowdCountingDataset(cfg.test_data_dir)

print("Train samples:", len(train_dataset))
print("Test samples:", len(test_dataset))

img, target = train_dataset[0]

print("Image shape:", img.shape)
print("Target shape:", target.shape)
print("Target count:", target.sum().item())