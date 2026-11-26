"""YOLOv9m person detector wrapper for the SafeCrowd AI pipeline.

Weights: YOLOv9m fine-tuned on WiderPerson (single class: person), trained at imgsz 960.
Hold-out test (500 images, NMS iou 0.6): mAP50 0.796, mAP50-95 0.489, P 0.808, R 0.687 (see README / eval_report.json).

Typical use (load ONCE at service start-up, reuse for every frame):

    detector = PersonDetector()                 # weights path from YOLO_WEIGHTS env var or ./weights/best_v2.pt
    people = detector.detect(frame_bgr)         # list[Detection]
    n = detector.count(frame_bgr)               # int

Requirements: pip install ultralytics==8.4.190   (the version the weights were trained with; newer 8.4.x also works)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "weights" / "best_v2.pt"


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float

    @property
    def xyxy(self):
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def center(self):
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


class PersonDetector:
    """Thin wrapper around ultralytics.YOLO with the settings recommended for this model.

    imgsz=960  : the model was fine-tuned at 960; 640 is faster but loses small / far persons.
    conf=0.25  : best-F1 threshold on val. Lower it (0.15-0.2) if you care more about recall / counting.
    iou=0.7    : NMS IoU. Crowded scenes need a high value so overlapping people are not merged.
    max_det    : upper limit of boxes per frame.
    """

    def __init__(self, weights=None, imgsz=960, conf=0.25, iou=0.7, max_det=1000, device=None, half=None):
        from ultralytics import YOLO  # imported here so that importing this module is cheap

        path = Path(weights or os.environ.get("YOLO_WEIGHTS", DEFAULT_WEIGHTS))
        if not path.is_file():
            raise FileNotFoundError(
                f"YOLO weights not found: {path}. Set the YOLO_WEIGHTS env var or pass weights=...")
        if half is None:
            import torch
            half = bool(torch.cuda.is_available())      # fp16 only makes sense on GPU
        self.weights, self.imgsz, self.conf, self.iou = path, imgsz, conf, iou
        self.max_det, self.device, self.half = max_det, device, half
        self.model = YOLO(str(path))

    def _predict(self, frame, conf=None):
        return self.model.predict(
            frame, imgsz=self.imgsz, conf=self.conf if conf is None else conf, iou=self.iou,
            max_det=self.max_det, classes=[0], device=self.device, half=self.half, verbose=False)[0]

    def detect(self, frame, conf=None):
        """frame: BGR numpy array (e.g. from cv2) or an image path. Returns list[Detection]."""
        res = self._predict(frame, conf)
        xyxy = res.boxes.xyxy.cpu().numpy()
        score = res.boxes.conf.cpu().numpy()
        return [Detection(float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(c)) for b, c in zip(xyxy, score)]

    def count(self, frame, conf=None):
        return len(self._predict(frame, conf).boxes)

    def annotate(self, frame, conf=None):
        """Returns a BGR image with the boxes drawn (for debugging / dashboards)."""
        return self._predict(frame, conf).plot(line_width=1, labels=False)
