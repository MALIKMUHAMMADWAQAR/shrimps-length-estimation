# Evaluation

Metrics and reported results reconstructed from Section 3 of the paper.

## Metrics

| Stage | Metrics | Implementation |
|-------|---------|----------------|
| Instance segmentation | AP50, AP75, AP[50:95], params, FPS | `src/evaluators/detection_metrics.py` |
| Centerline segmentation | Precision, Recall, F1, mIoU, params, FPS | `src/evaluators/segmentation_metrics.py` |
| Size estimation | MAE, RMSE (px & mm), FPS (Eqs. 10-11) | `src/evaluators/size_metrics.py` |

Run evaluation:

```bash
python scripts/evaluate.py --config configs/val.yaml --data-root datasets/centerline --checkpoint checkpoints/centerline_model.pth
# or a synthetic smoke evaluation
python scripts/evaluate.py --config configs/val.yaml --synthetic
```

## Reported results (paper)

### Table 1 — Instance segmentation

| Model | AP50 | AP75 | AP[50:95] | Params (M) | FPS |
|-------|------|------|-----------|------------|-----|
| SOLOv2 | 0.899 | 0.724 | 0.593 | 46.2 | 44 |
| YOLACT | 0.783 | 0.461 | 0.437 | 34.7 | 43 |
| CondInst | 0.934 | 0.663 | 0.574 | 37.5 | 42 |
| SparseInst | 0.836 | 0.428 | 0.432 | 42.6 | 43 |
| YOLOv8-m | 0.931 | 0.735 | 0.630 | 27.2 | 55 |
| **RTMDet-m** | **0.960** | **0.795** | **0.631** | 34.2 | **58** |

### Table 2 — Centerline segmentation (MobileNetV2 backbone)

| Model | Precision | Recall | F1 | mIoU | Param (M) | FPS |
|-------|-----------|--------|----|------|-----------|-----|
| UNet | 0.839 | 0.847 | 0.843 | 0.729 | 4.4 | 164 |
| UNet++ | 0.864 | 0.893 | 0.868 | 0.784 | 4.6 | 127 |
| LinkNet | 0.826 | 0.863 | 0.844 | 0.731 | 2.1 | 160 |
| DeepLabV3 | 0.842 | 0.882 | 0.860 | 0.754 | 2.1 | 169 |
| **Proposed** | **0.866** | **0.900** | **0.883** | **0.791** | 2.1 | 150 |

### Table 3 — Size estimation

| Model | MAE (px) | MAE (mm) | RMSE (px) | RMSE (mm) | FPS |
|-------|----------|----------|-----------|-----------|-----|
| PCFM | 84.76 | 44.61 | 105.95 | 55.76 | 10 |
| VDBM | 96.27 | 50.67 | 120.34 | 63.34 | 0.05 |
| UNet+SAG | 57.40 | 30.21 | 71.75 | 37.21 | 3.2 |
| **Our Framework** | **19.44** | **10.23** | **24.30** | **12.79** | 5.14 |

The proposed framework attains the lowest MAE (1.02 cm) and RMSE (1.27 cm).
