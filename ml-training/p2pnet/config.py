from dataclasses import dataclass
from typing import Optional


@dataclass
class P2PConfig:
    train_data_dir: str = "data/part_A_final/train_data"
    val_data_dir: str = "data/part_A_final/test_data"

    # PLACEHOLDER VALUES — verify against the team's actual P2PNet Colab
    # notebook before a full training run.
    epochs: int = 200
    batch_size: int = 8
    crop_size: int = 256
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4

    row: int = 2   # anchor-point grid per stride-8 cell: row x line
    line: int = 2  # 2x2 = 4 anchor points per cell (paper default)

    limit_samples: Optional[int] = None  # for quick sanity tests only

    checkpoint_dir: str = "checkpoints"
    save_every_n_epochs: int = 5

    seed: int = 42
    device: str = "cuda"
    num_workers: int = 2