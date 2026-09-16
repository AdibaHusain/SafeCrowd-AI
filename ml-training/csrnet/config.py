from dataclasses import dataclass


@dataclass
class CSRNetConfig:

    # Data
    train_data_dir: str = "processed_partA/train"
    test_data_dir: str = "processed_partA/test"

    # Training
    epochs: int = 200
    batch_size: int = 1

    learning_rate: float = 1e-6
    weight_decay: float = 5e-4

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every_n_epochs: int = 5

    # Misc
    seed: int = 42
    device: str = "cuda"