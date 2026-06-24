# Datasets

Two custom datasets are described in Section 2.2 of the paper.

## 1. Instance-segmentation dataset

- **1000** high-quality RGB images extracted from 3 environment video streams.
- Shrimp counts per environment: 10 / 15 / 18 (varying density), including
  occluded and non-occluded cases.
- Each shrimp is annotated with an instance mask and bounding box (COCO format).

Reader: `src/datasets/instance.py::CocoInstanceDataset`
(layout: an image directory + a COCO JSON annotation file).

## 2. Mask-to-centerline dataset

- ~**2400** shrimp mask instances, each manually annotated with a centerline.
- Built by training an instance segmenter on a small subset, predicting masks,
  filtering low-quality masks, then labelling centerlines.
- During training the centerline is rasterised as a **3-pixel-wide** band.

Reader: `src/datasets/centerline.py::CenterlineDataset`

Expected layout:

```
datasets/centerline/
  train/  {images,masks,centerlines}/<stem>.png
  val/    {images,masks,centerlines}/<stem>.png
  test/   {images,masks,centerlines}/<stem>.png
```

## Splits & tooling

- Split: **80 % train / 10 % val / 10 % test** for both datasets.
- Annotation tool: CVAT v2.30.0.
- Availability: the original data is available on request from the paper's
  corresponding author (see the Data Availability Statement).

## Imaging / experimental setup (Section 2.1)

- Shrimp length range: 4-11 cm.
- Container: 50 × 35 × 25 cm, water depth 3-5 cm.
- Camera: Intel RealSense D435, RGB mode, 30 FPS, 848 × 480 px, mounted 25 cm
  above the water surface.

## Synthetic data (this repository)

For development, testing and demonstration without the private dataset, the repo
provides a procedural generator (`src/datasets/synthetic.py`) producing curved
shrimp with ground-truth masks, centerlines and true lengths. It powers the unit
tests, the `--synthetic` training mode and the inference demo.
