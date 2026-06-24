"""Trainer for the centerline segmentation model (Section 3.2.1).

Training protocol from the paper: SGD optimizer, initial learning rate 0.02,
40 epochs, multi-step LR schedule, composite Dice + BCE loss (Equation 7).
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from ..evaluators.segmentation_metrics import SegmentationMetrics, SegmentationScores
from ..losses.segmentation import DiceBCELoss
from .base import BaseTrainer


class CenterlineTrainer(BaseTrainer):
    """Optimises a :class:`CenterlineSegmentationModel`."""

    def __init__(self, *args, loss_fn: Optional[torch.nn.Module] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn or DiceBCELoss(beta=1.0)

    def train_one_epoch(self, loader: DataLoader, epoch: int) -> float:
        """Run a single training epoch and return the mean loss."""
        self.model.train()
        running_loss = 0.0
        num_batches = 0
        for batch in loader:
            image = batch["image"].to(self.device, non_blocking=True)
            mask = batch["mask"].to(self.device, non_blocking=True)
            target = batch["target"].to(self.device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = self.model(image, mask)
                loss = self.loss_fn(logits, target)
            self._optimizer_step(loss)
            running_loss += float(loss.detach())
            num_batches += 1
        mean_loss = running_loss / max(num_batches, 1)
        self.logger.info("Epoch %d | train loss %.4f", epoch, mean_loss)
        return mean_loss

    @torch.no_grad()
    def validate(self, loader: DataLoader, threshold: float = 0.5) -> SegmentationScores:
        """Evaluate segmentation metrics on a validation loader."""
        self.model.eval()
        metrics = SegmentationMetrics(threshold=threshold)
        for batch in loader:
            image = batch["image"].to(self.device, non_blocking=True)
            mask = batch["mask"].to(self.device, non_blocking=True)
            target = batch["target"].to(self.device, non_blocking=True)
            logits = self.model(image, mask)
            metrics.update(logits, target)
        scores = metrics.compute()
        self.logger.info(
            "Validation | P %.4f R %.4f F1 %.4f mIoU %.4f",
            scores.precision,
            scores.recall,
            scores.f1,
            scores.iou,
        )
        return scores

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        epochs: int,
        save_name: str = "centerline_model.pth",
    ) -> Dict[str, float]:
        """Full training loop returning the final metrics."""
        best_f1 = 0.0
        history: Dict[str, float] = {}
        for epoch in range(1, epochs + 1):
            history["train_loss"] = self.train_one_epoch(train_loader, epoch)
            if val_loader is not None:
                scores = self.validate(val_loader)
                history.update(
                    {"precision": scores.precision, "recall": scores.recall, "f1": scores.f1, "miou": scores.iou}
                )
                if scores.f1 >= best_f1:
                    best_f1 = scores.f1
                    self.save(save_name, epoch=epoch, f1=scores.f1)
            if self.scheduler is not None:
                self.scheduler.step()
        if val_loader is None:
            self.save(save_name, epoch=epochs)
        return history
