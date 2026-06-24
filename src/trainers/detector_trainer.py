"""Trainer scaffolding for the RTMDet-ins instance-segmentation stage.

The paper trains the detector with the standard RTMDet recipe (Section 3.1.1):
SGD, learning rate 0.01, momentum 0.9, weight decay 1e-4, 70 epochs and a
multi-step LR schedule, using dynamic soft-label assignment.

A faithful re-implementation of RTMDet's dynamic label assignment is involved;
this trainer therefore exposes the architecture, loss (:class:`RTMDetLoss`) and
training hooks, and documents that production training should use the official
OpenMMLab RTMDet config with the datasets produced by :mod:`src.datasets`. The
single-step interface below is fully functional given a batch of pre-assigned
targets and is used by the unit tests to verify the loss/optimiser wiring.
"""

from __future__ import annotations

from typing import Dict

import torch

from ..losses.detection import RTMDetLoss
from .base import BaseTrainer


class DetectorTrainer(BaseTrainer):
    """Optimises an :class:`RTMDetIns` model given assigned targets."""

    def __init__(self, *args, alpha: float = 2.0, beta: float = 1.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.loss_fn = RTMDetLoss(alpha=alpha, beta=beta)

    def train_step(self, targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Run one optimisation step on a batch of pre-assigned targets.

        Args:
            targets: Dict with ``cls_pred``, ``cls_target``, ``box_pred``,
                ``box_target``, ``mask_pred`` and ``mask_target`` tensors.

        Returns:
            A dictionary of scalar loss values.
        """
        self.model.train()
        with torch.amp.autocast("cuda", enabled=self.use_amp):
            losses = self.loss_fn(
                targets["cls_pred"],
                targets["cls_target"],
                targets["box_pred"],
                targets["box_target"],
                targets["mask_pred"],
                targets["mask_target"],
            )
        self._optimizer_step(losses["loss_total"])
        return {key: float(value.detach()) for key, value in losses.items()}
