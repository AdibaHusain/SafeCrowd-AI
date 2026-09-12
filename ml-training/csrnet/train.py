import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import CSRNetConfig
from model import CSRNet
from dataset import CrowdCountingDataset


def validate(model, val_loader, device):
    model.eval()
    mae = 0.0
    with torch.no_grad():
        for img, target in val_loader:
            img, target = img.to(device), target.to(device)
            pred = model(img)
            mae += abs(pred.sum().item() - target.sum().item())
    model.train()
    return mae / len(val_loader)


def main():
    cfg = CSRNetConfig()
    torch.manual_seed(cfg.seed)

    device = cfg.device if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    train_dataset = CrowdCountingDataset(cfg.train_data_dir)
    val_dataset = CrowdCountingDataset(cfg.val_data_dir)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

    model = CSRNet().to(device)
    criterion = nn.MSELoss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    best_mae = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        epoch_loss = 0.0
        for img, target in train_loader:
            img, target = img.to(device), target.to(device)

            optimizer.zero_grad()
            pred = model(img)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch}/{cfg.epochs} — loss: {avg_loss:.4f}")

        if epoch % cfg.save_every_n_epochs == 0 or epoch == cfg.epochs:
            val_mae = validate(model, val_loader, device)
            print(f"  Validation MAE: {val_mae:.2f}")

            if val_mae < best_mae:
                best_mae = val_mae
                checkpoint_path = os.path.join(cfg.checkpoint_dir, "csrnet_best.pth")
                torch.save(model.state_dict(), checkpoint_path)
                print(f"  New best model saved -> {checkpoint_path} (MAE {best_mae:.2f})")

    print(f"Training complete. Best validation MAE: {best_mae:.2f}")


if __name__ == "__main__":
    main()