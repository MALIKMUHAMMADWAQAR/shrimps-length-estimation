"""Semantic-segmentation metrics (precision, recall, F1, mIoU).

Used to evaluate the centerline predictive module (Table 2).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SegmentationScores:
    """Aggregated binary segmentation metrics."""

    precision: float
    recall: float
    f1: float
    iou: float


class SegmentationMetrics:
    """Accumulates true/false positives/negatives over a dataset."""

    def __init__(self, threshold: float = 0.5, eps: float = 1e-7) -> None:
        self.threshold = threshold
        self.eps = eps
        self.reset()

    def reset(self) -> None:
        """Zero all running counters."""
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0
        self.tn = 0.0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        """Update counters from a batch of logits and binary targets."""
        pred = (torch.sigmoid(logits) >= self.threshold).float()
        target = (target >= 0.5).float()
        self.tp += float((pred * target).sum())
        self.fp += float((pred * (1 - target)).sum())
        self.fn += float(((1 - pred) * target).sum())
        self.tn += float(((1 - pred) * (1 - target)).sum())

    def compute(self) -> SegmentationScores:
        """Return precision, recall, F1 and IoU over all updates."""
        precision = self.tp / (self.tp + self.fp + self.eps)
        recall = self.tp / (self.tp + self.fn + self.eps)
        f1 = 2 * precision * recall / (precision + recall + self.eps)
        iou = self.tp / (self.tp + self.fp + self.fn + self.eps)
        return SegmentationScores(precision=precision, recall=recall, f1=f1, iou=iou)
