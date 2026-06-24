"""Detection / instance-segmentation losses for RTMDet (Equation 3).

``L_total = sum_i FL(cls) + alpha * GIoU(box) + beta * BCE(mask)``.

The individual terms are implemented as reusable modules so the same loss can be
used both with the bundled from-scratch RTMDet and with an external trainer.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Sigmoid focal loss for classification (Lin et al., 2017)."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the focal loss between logits and multi-hot targets."""
        prob = torch.sigmoid(pred)
        ce = F.binary_cross_entropy_with_logits(pred, target.to(pred.dtype), reduction="none")
        p_t = prob * target + (1 - prob) * (1 - target)
        loss = ce * ((1 - p_t) ** self.gamma)
        if self.alpha >= 0:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            loss = alpha_t * loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Element-wise Generalized IoU between two ``(N, 4)`` xyxy box tensors."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    lt = torch.max(boxes1[:, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    union = area1 + area2 - inter
    iou = inter / union.clamp(min=1e-7)

    enclose_lt = torch.min(boxes1[:, :2], boxes2[:, :2])
    enclose_rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])
    enclose_wh = (enclose_rb - enclose_lt).clamp(min=0)
    enclose_area = enclose_wh[:, 0] * enclose_wh[:, 1]
    giou = iou - (enclose_area - union) / enclose_area.clamp(min=1e-7)
    return giou


class GIoULoss(nn.Module):
    """Generalized IoU regression loss."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Return ``1 - GIoU`` reduced over the matched boxes."""
        loss = 1.0 - generalized_box_iou(pred_boxes, target_boxes)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class MaskBCELoss(nn.Module):
    """Pixel-wise binary cross-entropy mask loss."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, pred_masks: torch.Tensor, target_masks: torch.Tensor) -> torch.Tensor:
        """BCE between predicted and ground-truth instance masks."""
        return F.binary_cross_entropy_with_logits(
            pred_masks, target_masks.to(pred_masks.dtype), reduction=self.reduction
        )


class RTMDetLoss(nn.Module):
    """Weighted sum of the RTMDet classification, box and mask losses (Eq. 3).

    Args:
        alpha: Weight on the GIoU box-regression term.
        beta: Weight on the mask BCE term.
    """

    def __init__(self, alpha: float = 2.0, beta: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.focal = FocalLoss()
        self.giou = GIoULoss()
        self.mask_bce = MaskBCELoss()

    def forward(
        self,
        cls_pred: torch.Tensor,
        cls_target: torch.Tensor,
        box_pred: torch.Tensor,
        box_target: torch.Tensor,
        mask_pred: torch.Tensor,
        mask_target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return a dict with the individual and total losses."""
        loss_cls = self.focal(cls_pred, cls_target)
        loss_box = self.giou(box_pred, box_target)
        loss_mask = self.mask_bce(mask_pred, mask_target)
        total = loss_cls + self.alpha * loss_box + self.beta * loss_mask
        return {
            "loss_cls": loss_cls,
            "loss_box": loss_box,
            "loss_mask": loss_mask,
            "loss_total": total,
        }
