"""End-to-end tests: datasets, training step and the full framework."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from src.datasets import SyntheticCenterlineDataset, generate_scene
from src.models import (
    CenterlineSegmentationModel,
    DualSegmentationFramework,
    FrameworkConfig,
    RTMDetIns,
)
from src.trainers import CenterlineTrainer


def test_synthetic_dataset_item_shapes():
    dataset = SyntheticCenterlineDataset(length=4, size=(64, 64))
    sample = dataset[0]
    assert sample["image"].shape == (3, 64, 64)
    assert sample["mask"].shape == (1, 64, 64)
    assert sample["target"].shape == (1, 64, 64)
    assert sample["target"].max() <= 1.0


def test_centerline_training_reduces_loss():
    dataset = SyntheticCenterlineDataset(length=8, size=(64, 64))
    loader = DataLoader(dataset, batch_size=4)
    model = CenterlineSegmentationModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    trainer = CenterlineTrainer(model, optimizer, torch.device("cpu"))
    first = trainer.train_one_epoch(loader, epoch=1)
    last = first
    for epoch in range(2, 5):
        last = trainer.train_one_epoch(loader, epoch=epoch)
    assert last <= first  # loss should not increase over a few epochs


def test_generate_scene_consistency():
    rgb, masks, centerlines, lengths = generate_scene(128, 128, num_shrimp=3, seed=1)
    assert rgb.shape == (128, 128, 3)
    assert len(masks) == len(centerlines) == len(lengths) == 3


def test_full_framework_runs():
    detector = RTMDetIns(num_classes=1).eval()
    centerline = CenterlineSegmentationModel().eval()
    framework = DualSegmentationFramework(
        detector, centerline, FrameworkConfig(score_threshold=0.0, max_instances=3, pixel_to_mm=0.5)
    )
    image = torch.rand(1, 3, 128, 128)
    results = framework(image)
    assert len(results) == 1
    for measurement in results[0]:
        assert measurement.length_mm >= 0.0
        assert measurement.centerline.ndim == 2
