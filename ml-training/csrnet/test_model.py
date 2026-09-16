import torch

from config import CSRNetConfig
from dataset import CrowdCountingDataset
from model import CSRNet


# Load configuration
cfg = CSRNetConfig()

# Select device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

# Load dataset
train_dataset = CrowdCountingDataset(cfg.train_data_dir)

# Get one sample
img, target = train_dataset[0]

# Add batch dimension
img = img.unsqueeze(0).to(device)

print("Input shape:", img.shape)

# Create model
model = CSRNet(load_pretrained_vgg=False).to(device)

# Evaluation mode
model.eval()

# Forward pass without calculating gradients
with torch.no_grad():
    output = model(img)

print("Output shape:", output.shape)
print("Output minimum:", output.min().item())
print("Output maximum:", output.max().item())
print("Predicted count:", output.sum().item())
print("Actual count:", target.sum().item())