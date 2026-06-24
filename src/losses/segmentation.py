"""Segmentation losses for the centerline model (Equations 7-8).

The centerline model is optimised with a composite of Dice loss and Binary
Cross-Entropy: ``L_seg = L_Dice + beta * L_BCE``. Dice loss combats the heavy
foreground/background imbalance of thin centerline structures, while BCE
penalises per-pixel misclassification.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation.

    Args:
        eps: Numerical stability constant (``epsilon`` in Equation 8).
        from_logits: If ``True`` a sigmoid is applied to the predictions first.
    """

    def __init__(self, eps: float = 1e-6, from_logits: bool = True) -> None:
        super().__init__()
        self.eps = eps
        self.from_logits = from_logits

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return ``1 - Dice`` averaged over the batch."""
        if self.from_logits:
            pred = torch.sigmoid(pred)
        pred = pred.reshape(pred.size(0), -1)
        target = target.reshape(target.size(0), -1).to(pred.dtype)
        intersection = (pred * target).sum(dim=1)
        denominator = pred.sum(dim=1) + target.sum(dim=1)
        dice = (2.0 * intersection + self.eps) / (denominator + self.eps)
        return (1.0 - dice).mean()


class DiceBCELoss(nn.Module):
    """Composite Dice + BCE loss (Equation 7).

    Args:
        beta: Weight ``beta`` on the BCE term.
        eps: Numerical stability constant.
        pos_weight: Optional positive-class weight passed to BCE.
    """

    def __init__(self, beta: float = 1.0, eps: float = 1e-6, pos_weight: float | None = None) -> None:
        super().__init__()
        self.beta = beta
        self.dice = DiceLoss(eps=eps, from_logits=True)
        self.register_buffer(
            "pos_weight",
            torch.tensor([pos_weight]) if pos_weight is not None else None,
            persistent=False,
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Combine Dice and (weighted) BCE losses."""
        target = target.to(pred.dtype)
        bce = F.binary_cross_entropy_with_logits(pred, target, pos_weight=self.pos_weight)
        dice = self.dice(pred, target)
        return dice + self.beta * bce
