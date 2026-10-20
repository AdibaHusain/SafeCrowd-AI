
from dataclasses import dataclass
from pathlib import Path


@dataclass
class YOLOConfig:
    data_yaml: str = str(
        Path(__file__).resolve().parent / "data" / "data.yaml"
    )
    pretrained_weights: str = "yolov9m.pt"
    epochs: int = 10
    batch_size: int = 4
    img_size: int = 640
    project_dir: str = str(
        Path(__file__).resolve().parent / "runs"
    )
    run_name: str = "yolov9m_person"
    device: str = "cpu"
