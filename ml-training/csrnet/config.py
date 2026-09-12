from dataclasses import dataclass


@dataclass
class CSRNetConfig:
    # Data
    train_data_dir: str = "data/shanghaitech_partA/train"
    val_data_dir: str = "data/shanghaitech_partA/val"

    # Training
    epochs: int = 200
    batch_size: int = 1          # CSRNet trains on full-resolution images -> batch size 1 is standard
    learning_rate: float = 1e-6
    weight_decay: float = 5e-4

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every_n_epochs: int = 5

    # Misc
    seed: int = 42
    device: str = "cuda"  # falls back to cpu automatically in train.py if unavailable