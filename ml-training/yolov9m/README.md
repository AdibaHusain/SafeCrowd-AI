# YOLOv9m person detector v2

Two-stage fine-tune of COCO-pretrained YOLOv9m on WiderPerson (single class: person).
Stage 1: imgsz 640, 20 epochs. Stage 2: imgsz 960, 20 epochs.
Train set: <= 150 persons per image, ignore/crowd regions painted grey. Val and hold-out test are unfiltered.

| Model | Split | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| v1 (640) | val | 0.827 | 0.698 | 0.792 | 0.495 |
| v1 (640) | hold-out test | 0.804 | 0.681 | 0.774 | 0.475 |
| v2 (960) | val | 0.820 | 0.718 | 0.813 | 0.515 |
| v2 (960) | hold-out test | 0.808 | 0.687 | 0.796 | 0.489 |
| v2 (960), tuned NMS | val | 0.823 | 0.709 | 0.809 | 0.519 |
| v2 (960), tuned NMS | hold-out test | 0.814 | 0.682 | 0.793 | 0.493 |

Recommended inference settings: conf 0.25, NMS iou 0.7, max_det 1000.
The v1 rows and the plain v2 rows use the same protocol (NMS iou 0.6); the tuned rows use the settings above.
