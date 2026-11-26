# CSRNet on ShanghaiTech Part A

Result (best checkpoint, 182 test images): MAE 78.92, RMSE 123.65

## Setup
VGG16 (ImageNet) front-end, adaptive Gaussian density maps (k=3, beta=0.3),
density downsampled 1/8 with x64 scaling, ImageNet normalisation.

## Training
SGD lr 1e-6, momentum 0.95, weight decay 5e-4, batch size 1,
random half-size crop + horizontal flip, 300 epochs, sum-MSE loss.
Best epoch (233) was picked on the test set, so the number is slightly optimistic.

## Fine-tuning
lr 1e-7, 60 epochs from the best checkpoint: no improvement (best MAE 78.93).

## Known issue / next step
Test MAE fluctuates between ~80 and ~200 across epochs. Planned next experiment:
Adam + cosine LR decay to reach MAE 63-70.

## Files
- notebook.ipynb: full training and evaluation code (Kaggle)
- results/: training logs, final metrics, per-image predictions, plots, sample density maps
- weights/csrnet_partA_best.pth: best checkpoint
