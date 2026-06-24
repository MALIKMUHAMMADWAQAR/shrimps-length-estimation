"""Geometric helpers for size estimation from a skeletonised centerline."""

from __future__ import annotations

import math

import numpy as np

# Euclidean cost of a step to each of the 8 neighbours.
_NEIGHBOUR_OFFSETS = [
    (-1, -1, math.sqrt(2)),
    (-1, 0, 1.0),
    (-1, 1, math.sqrt(2)),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (1, -1, math.sqrt(2)),
    (1, 0, 1.0),
    (1, 1, math.sqrt(2)),
]


def centerline_length_pixels(skeleton: np.ndarray) -> float:
    """Estimate the arc length (in pixels) of a one-pixel-wide skeleton.

    The skeleton is treated as a graph where 4-connected neighbours contribute a
    cost of 1 and 8-connected (diagonal) neighbours contribute ``sqrt(2)``. The
    total arc length is half the sum of all edge costs (each internal edge is
    counted from both endpoints).

    Args:
        skeleton: 2-D binary array with skeleton pixels set to non-zero.

    Returns:
        Estimated arc length in pixels (``0.0`` for an empty skeleton).
    """
    sk = (np.asarray(skeleton) > 0).astype(np.uint8)
    if sk.sum() == 0:
        return 0.0
    if sk.sum() == 1:
        return 1.0

    padded = np.pad(sk, 1, mode="constant", constant_values=0)
    total = 0.0
    for dy, dx, cost in _NEIGHBOUR_OFFSETS:
        shifted = padded[1 + dy : 1 + dy + sk.shape[0], 1 + dx : 1 + dx + sk.shape[1]]
        total += cost * float(np.sum((sk == 1) & (shifted == 1)))
    return total / 2.0


def pixels_to_mm(length_px: float, pixel_to_mm: float) -> float:
    """Convert a pixel length to millimetres using a calibration factor."""
    return length_px * pixel_to_mm
