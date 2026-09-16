import os
import torch
from torch.utils.data import DataLoader

from config import P2PConfig
from model import P2PNet
from dataset import P2PDataset, collate_fn
from matcher import P2PLoss


def validate(model, val_loader, device):
    model.eval()
    mae = 0.0
    with torch.no_grad():
        for imgs, points_list in val_loader:
            imgs = imgs.to(device)
            _, cls_logits = model(imgs)
            for b in range(imgs.shape[0]):
                pred_count = (cls_logits[b].argmax(dim=-1) == 1).sum().item()
                gt_count = points_list[b].shape[0]
                mae += abs(pred_count - gt_count)
    model.train()
    return mae / len(val_loader.dataset)


def main():
    cfg = P2PConfig()
    torch.manual_seed(cfg.seed)
    device = cfg.device if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    train_dataset = P2PDataset(
        cfg.train_data_dir, train=True, crop_size=cfg.crop_size, limit_samples=cfg.limit_samples
    )
    val_dataset = P2PDataset(
        cfg.val_data_dir, train=False, limit_samples=cfg.limit_samples
    )

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=collate_fn,
    )

    model = P2PNet(row=cfg.row, line=cfg.line).to(device)
    criterion = P2PLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    best_mae = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        epoch_loss = 0.0
        for imgs, points_list in train_loader:
            imgs = imgs.to(device)
            points_list = [p.to(device) for p in points_list]

            optimizer.zero_grad()
            pred_points, cls_logits = model(imgs)
            loss, cls_l, reg_l = criterion(pred_points, cls_logits, points_list)
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
                path = os.path.join(cfg.checkpoint_dir, "p2pnet_best.pth")
                torch.save(model.state_dict(), path)
                print(f"  New best model saved -> {path} (MAE {best_mae:.2f})")

    print(f"Training complete. Best validation MAE: {best_mae:.2f}")


if __name__ == "__main__":
    main()