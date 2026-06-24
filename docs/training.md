# Training

Training protocols reconstructed from Section 3.1.1 (instance segmentation) and
Section 3.2.1 (centerline) of the paper.

## Stage 2 — Centerline segmentation model (bundled, fully runnable)

| Hyper-parameter | Value (paper) |
|-----------------|---------------|
| Optimizer       | SGD |
| Initial LR      | 0.02 |
| LR schedule     | Multi-step (decay) |
| Epochs          | 40 |
| Encoder backbone| MobileNetV2 |
| Loss            | `L_seg = L_Dice + β · L_BCE` (Eqs. 7-8) |
| GT centerline   | 3-pixel-wide band |
| Hardware        | NVIDIA TITAN RTX (24 GB) |

Run training (real data):

```bash
python scripts/train.py --config configs/train.yaml --data-root datasets/centerline
```

Smoke run on procedurally generated shrimp (no data required):

```bash
python scripts/train.py --config configs/train.yaml --synthetic --epochs 2
```

Mixed precision and multi-GPU:

```bash
# AMP (CUDA only)
python scripts/train.py --config configs/train.yaml --synthetic --amp
# Multi-GPU via torchrun
torchrun --nproc_per_node=2 scripts/train.py --config configs/train.yaml --data-root datasets/centerline
```

## Stage 1 — RTMDet-ins-m instance segmentation

| Hyper-parameter | Value (paper) |
|-----------------|---------------|
| Optimizer       | SGD |
| LR / momentum / weight decay | 0.01 / 0.9 / 1e-4 |
| LR schedule     | Multi-step |
| Epochs          | 70 |
| Backbone        | CSPNeXt |
| Loss            | `L_total = FL(cls) + α · GIoU(box) + β · BCE(mask)` (Eq. 3) |
| Hardware        | NVIDIA TITAN RTX (24 GB) |

The bundled `RTMDetIns` and `DetectorTrainer` expose the exact architecture and
losses. Because RTMDet's dynamic soft-label assignment is intricate, the
recommended path for reproducing the reported AP is to train the official
OpenMMLab RTMDet-ins-m config on the COCO-format dataset emitted by
`src/datasets/instance.py`. `DetectorTrainer.train_step` is provided for
loss/optimiser verification and custom assignment experiments.
