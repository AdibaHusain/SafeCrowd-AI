import os
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from config import CSRNetConfig
from model import CSRNet
from dataset import CrowdCountingDataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_mae(model, data_loader, device):
    """
    Calculate Mean Absolute Error using predicted and actual crowd counts.
    """
    model.eval()
    total_absolute_error = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            predictions = model(images)

            predicted_counts = predictions.sum(dim=(1, 2, 3))
            actual_counts = targets.sum(dim=(1, 2, 3))

            absolute_error = torch.abs(
                predicted_counts - actual_counts
            )

            total_absolute_error += absolute_error.sum().item()
            total_samples += images.size(0)

    model.train()

    if total_samples == 0:
        return float("inf")

    return total_absolute_error / total_samples


def main():
    cfg = CSRNetConfig()
    set_seed(cfg.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print("CSRNet Training")
    print("=" * 60)
    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Epochs: {cfg.epochs}")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Learning rate: {cfg.learning_rate}")
    print("=" * 60)

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------
    full_train_dataset = CrowdCountingDataset(
        cfg.train_data_dir
    )

    test_dataset = CrowdCountingDataset(
        cfg.test_data_dir
    )

    # Split original training data into train and validation.
    validation_size = int(0.10 * len(full_train_dataset))
    training_size = len(full_train_dataset) - validation_size

    train_dataset, validation_dataset = random_split(
        full_train_dataset,
        [training_size, validation_size],
        generator=torch.Generator().manual_seed(cfg.seed)
    )

    print(f"Total training samples: {len(full_train_dataset)}")
    print(f"Training split: {len(train_dataset)}")
    print(f"Validation split: {len(validation_dataset)}")
    print(f"Final test samples: {len(test_dataset)}")

    # ---------------------------------------------------------
    # DataLoaders
    # ---------------------------------------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------
    model = CSRNet(load_pretrained_vgg=True).to(device)

    criterion = nn.MSELoss(reduction="sum")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay
    )

    # ---------------------------------------------------------
    # Checkpoint directory
    # ---------------------------------------------------------
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    best_validation_mae = float("inf")
    history = []

    # ---------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            predictions = model(images)
            loss = criterion(predictions, targets)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        average_loss = epoch_loss / len(train_loader)

        # Evaluate on validation split every epoch.
        validation_mae = calculate_mae(
            model,
            validation_loader,
            device
        )

        history.append({
            "epoch": epoch,
            "loss": average_loss,
            "validation_mae": validation_mae
        })

        print(
            f"Epoch [{epoch:03d}/{cfg.epochs}] | "
            f"Loss: {average_loss:.4f} | "
            f"Validation MAE: {validation_mae:.4f}"
        )

        # Save best model according to validation MAE.
        if validation_mae < best_validation_mae:
            best_validation_mae = validation_mae

            best_checkpoint_path = os.path.join(
                cfg.checkpoint_dir,
                "csrnet_best.pth"
            )

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "validation_mae": validation_mae,
                "config": cfg.__dict__
            }, best_checkpoint_path)

            print(
                f"  Best checkpoint saved: "
                f"{best_checkpoint_path}"
            )

        # Save periodic checkpoint.
        if (
            epoch % cfg.save_every_n_epochs == 0
            or epoch == cfg.epochs
        ):
            checkpoint_path = os.path.join(
                cfg.checkpoint_dir,
                f"csrnet_epoch_{epoch:03d}.pth"
            )

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "validation_mae": validation_mae,
                "config": cfg.__dict__
            }, checkpoint_path)

            print(
                f"  Periodic checkpoint saved: "
                f"{checkpoint_path}"
            )

    # ---------------------------------------------------------
    # Final evaluation on untouched test set
    # ---------------------------------------------------------
    best_checkpoint_path = os.path.join(
        cfg.checkpoint_dir,
        "csrnet_best.pth"
    )

    if os.path.exists(best_checkpoint_path):
        checkpoint = torch.load(
            best_checkpoint_path,
            map_location=device
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        final_test_mae = calculate_mae(
            model,
            test_loader,
            device
        )

        print("=" * 60)
        print("Final Evaluation")
        print("=" * 60)
        print(f"Best validation MAE: {best_validation_mae:.4f}")
        print(f"Final test MAE: {final_test_mae:.4f}")
        print("=" * 60)

        np.save(
            os.path.join(cfg.checkpoint_dir, "training_history.npy"),
            np.array(history, dtype=object)
        )

        with open(
            os.path.join(cfg.checkpoint_dir, "final_metrics.txt"),
            "w"
        ) as file:
            file.write(
                f"Best validation MAE: {best_validation_mae:.4f}\n"
            )
            file.write(
                f"Final test MAE: {final_test_mae:.4f}\n"
            )

    print("Training completed successfully.")


if __name__ == "__main__":
    main()
