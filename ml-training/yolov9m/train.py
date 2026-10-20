import torch
from pathlib import Path
from ultralytics import YOLO
from config import YOLOConfig


def main():
    cfg = YOLOConfig()

    device = cfg.device if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
    print(f"Dataset: {cfg.data_yaml}")
    print(f"Epochs: {cfg.epochs}")

    model = YOLO(cfg.pretrained_weights)

    results = model.train(
        data=cfg.data_yaml,
        epochs=cfg.epochs,
        batch=cfg.batch_size,
        imgsz=cfg.img_size,
        device=device,
        project=cfg.project_dir,
        name=cfg.run_name,
        workers=2,
    )

    save_dir = Path(results.save_dir)
    best_weights = save_dir / "weights" / "best.pt"

    print(f"Training completed. Best weights: {best_weights}")


if __name__ == "__main__":
    main()