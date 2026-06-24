"""Detection / instance-segmentation Average Precision metrics (Table 1).

A small, self-contained AP implementation (bounding-box IoU based) is provided so
the package can compute AP50 / AP75 / AP[50:95] without external dependencies.
For the exact COCO protocol used in the paper, ``pycocotools`` can be plugged in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


def _iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of xyxy boxes."""
    if boxes_a.size == 0 or boxes_b.size == 0:
        return np.zeros((boxes_a.shape[0], boxes_b.shape[0]))
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    lt = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    rb = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-7, None)


@dataclass
class DetectionScores:
    """COCO-style average precision summary."""

    ap50: float
    ap75: float
    ap50_95: float


def _average_precision_at(
    pred_boxes: Sequence[np.ndarray],
    pred_scores: Sequence[np.ndarray],
    gt_boxes: Sequence[np.ndarray],
    iou_threshold: float,
) -> float:
    """Compute AP at a single IoU threshold over a set of images."""
    all_scores: List[float] = []
    all_tp: List[int] = []
    num_gt = 0
    for boxes, scores, gts in zip(pred_boxes, pred_scores, gt_boxes):
        num_gt += len(gts)
        if len(boxes) == 0:
            continue
        order = np.argsort(-scores)
        boxes, scores = boxes[order], scores[order]
        matched = np.zeros(len(gts), dtype=bool)
        ious = _iou_matrix(boxes, gts) if len(gts) else np.zeros((len(boxes), 0))
        for i in range(len(boxes)):
            all_scores.append(float(scores[i]))
            if ious.shape[1] == 0:
                all_tp.append(0)
                continue
            best = int(np.argmax(ious[i]))
            if ious[i, best] >= iou_threshold and not matched[best]:
                matched[best] = True
                all_tp.append(1)
            else:
                all_tp.append(0)

    if num_gt == 0 or not all_scores:
        return 0.0
    order = np.argsort(-np.asarray(all_scores))
    tp = np.asarray(all_tp)[order]
    fp = 1 - tp
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / num_gt
    precision = tp_cum / np.clip(tp_cum + fp_cum, 1e-7, None)

    # 101-point interpolation (COCO style).
    recall_levels = np.linspace(0, 1, 101)
    ap = 0.0
    for level in recall_levels:
        prec = precision[recall >= level]
        ap += (prec.max() if prec.size else 0.0) / 101
    return float(ap)


def compute_detection_metrics(
    pred_boxes: Sequence[np.ndarray],
    pred_scores: Sequence[np.ndarray],
    gt_boxes: Sequence[np.ndarray],
) -> DetectionScores:
    """Compute AP50, AP75 and AP[50:95] over a dataset."""
    ap50 = _average_precision_at(pred_boxes, pred_scores, gt_boxes, 0.50)
    ap75 = _average_precision_at(pred_boxes, pred_scores, gt_boxes, 0.75)
    thresholds = np.arange(0.5, 1.0, 0.05)
    ap_all = np.mean(
        [_average_precision_at(pred_boxes, pred_scores, gt_boxes, t) for t in thresholds]
    )
    return DetectionScores(ap50=ap50, ap75=ap75, ap50_95=float(ap_all))
