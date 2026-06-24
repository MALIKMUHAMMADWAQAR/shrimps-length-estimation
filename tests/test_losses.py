"""Tests for the loss functions."""

from __future__ import annotations

import torch

from src.losses import (
    DiceBCELoss,
    DiceLoss,
    FocalLoss,
    GIoULoss,
    RTMDetLoss,
    generalized_box_iou,
)


def test_dice_loss_perfect_prediction_is_low():
    target = torch.zeros(1, 1, 8, 8)
    target[..., 2:6, 2:6] = 1
    logits = (target * 20) - 10  # strongly correct logits
    loss = DiceLoss()(logits, target)
    assert loss.item() < 0.05


def test_dice_bce_positive():
    loss = DiceBCELoss(beta=1.0)
    value = loss(torch.randn(2, 1, 16, 16), torch.randint(0, 2, (2, 1, 16, 16)).float())
    assert value.item() > 0


def test_focal_loss_runs():
    pred = torch.randn(4, 1)
    target = torch.randint(0, 2, (4, 1)).float()
    assert FocalLoss()(pred, target).item() >= 0


def test_giou_identical_boxes():
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    assert torch.allclose(generalized_box_iou(boxes, boxes), torch.ones(1), atol=1e-5)
    assert GIoULoss()(boxes, boxes).item() < 1e-5


def test_rtmdet_loss_keys():
    losses = RTMDetLoss()(
        torch.randn(3, 1),
        torch.randint(0, 2, (3, 1)).float(),
        torch.tensor([[0.0, 0.0, 5.0, 5.0]] * 3),
        torch.tensor([[0.0, 0.0, 5.0, 5.0]] * 3),
        torch.randn(3, 8, 8),
        torch.randint(0, 2, (3, 8, 8)).float(),
    )
    assert set(losses) == {"loss_cls", "loss_box", "loss_mask", "loss_total"}
    assert losses["loss_total"].item() >= 0
