"""Tests for skeletonization, geometry and metrics utilities."""

from __future__ import annotations

import numpy as np

from src.evaluators import compute_detection_metrics, compute_size_metrics
from src.utils import centerline_length_pixels, compute_pixel_to_mm, zhang_suen_thinning


def test_zhang_suen_produces_one_pixel_line():
    img = np.zeros((20, 20), dtype=np.uint8)
    img[8:12, 2:18] = 1  # a thick horizontal bar
    skeleton = zhang_suen_thinning(img)
    # Every column in the spanned range should keep at most ~1 pixel of thickness.
    column_counts = skeleton[:, 5:15].sum(axis=0)
    assert column_counts.max() <= 2
    assert skeleton.sum() > 0


def test_centerline_length_horizontal():
    skeleton = np.zeros((10, 10), dtype=np.uint8)
    skeleton[5, 1:9] = 1  # 8 pixels -> 7 unit steps
    length = centerline_length_pixels(skeleton)
    assert abs(length - 7.0) < 1e-6


def test_centerline_length_diagonal():
    skeleton = np.zeros((10, 10), dtype=np.uint8)
    for i in range(1, 6):
        skeleton[i, i] = 1  # 5 pixels diagonally -> 4 * sqrt(2)
    length = centerline_length_pixels(skeleton)
    assert abs(length - 4 * np.sqrt(2)) < 1e-6


def test_size_metrics():
    scores = compute_size_metrics([10.0, 12.0], [11.0, 10.0])
    assert abs(scores.mae - 1.5) < 1e-6
    assert scores.rmse >= scores.mae


def test_pixel_to_mm():
    assert abs(compute_pixel_to_mm(100.0, 50.0) - 0.5) < 1e-9


def test_detection_metrics_perfect():
    boxes = [np.array([[0, 0, 10, 10]], dtype=float)]
    scores = [np.array([0.9])]
    gts = [np.array([[0, 0, 10, 10]], dtype=float)]
    result = compute_detection_metrics(boxes, scores, gts)
    assert result.ap50 > 0.99
