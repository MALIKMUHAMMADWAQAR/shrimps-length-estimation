"""Synthetic shrimp data generation.

Provides procedurally generated curved "shrimp" (a body mask plus its true
centerline) so the full framework can be trained, tested and demonstrated
without access to the paper's private dataset. The generator is deterministic
given a seed, which makes it convenient for unit tests and smoke runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .transforms import dilate_binary


def _quadratic_bezier(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, num: int = 200) -> np.ndarray:
    """Sample ``num`` points along a quadratic Bezier curve."""
    t = np.linspace(0.0, 1.0, num)[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2


def _stamp_disks(canvas: np.ndarray, points: np.ndarray, radius: float) -> None:
    """Paint filled disks of ``radius`` at each point onto a binary canvas."""
    h, w = canvas.shape
    r = int(np.ceil(radius))
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    disk = (xx**2 + yy**2) <= radius**2
    for px, py in points:
        cx, cy = int(round(px)), int(round(py))
        x0, x1 = max(0, cx - r), min(w, cx + r + 1)
        y0, y1 = max(0, cy - r), min(h, cy + r + 1)
        dx0, dy0 = x0 - (cx - r), y0 - (cy - r)
        canvas[y0:y1, x0:x1] |= disk[dy0 : dy0 + (y1 - y0), dx0 : dx0 + (x1 - x0)]


def generate_shrimp(
    height: int,
    width: int,
    rng: np.random.Generator,
    body_radius: Tuple[float, float] = (4.0, 7.0),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Generate one curved shrimp.

    Returns:
        ``(rgb, mask, centerline, true_length_px)`` where ``rgb`` is ``uint8``
        ``HxWx3`` and ``mask`` / ``centerline`` are ``uint8`` ``HxW``.
    """
    margin = 0.18
    p0 = np.array([rng.uniform(margin, 1 - margin) * width, rng.uniform(margin, 1 - margin) * height])
    p2 = np.array([rng.uniform(margin, 1 - margin) * width, rng.uniform(margin, 1 - margin) * height])
    mid = (p0 + p2) / 2
    normal = np.array([-(p2 - p0)[1], (p2 - p0)[0]])
    norm = np.linalg.norm(normal) + 1e-6
    curvature = rng.uniform(-0.45, 0.45) * np.linalg.norm(p2 - p0)
    p1 = mid + normal / norm * curvature

    curve = _quadratic_bezier(p0, p1, p2, num=240)
    true_length = float(np.sum(np.linalg.norm(np.diff(curve, axis=0), axis=1)))

    centerline = np.zeros((height, width), dtype=np.uint8)
    _stamp_disks(centerline, curve, radius=0.5)

    radius = rng.uniform(*body_radius)
    mask = np.zeros((height, width), dtype=np.uint8)
    _stamp_disks(mask, curve, radius=radius)

    # Render a translucent shrimp on a noisy background to mimic the imaging setup.
    background = rng.integers(150, 210, size=(height, width, 3), dtype=np.uint8)
    body_colour = rng.integers(80, 150, size=3, dtype=np.uint8)
    rgb = background.copy()
    alpha = 0.55
    rgb[mask > 0] = (alpha * body_colour + (1 - alpha) * rgb[mask > 0]).astype(np.uint8)
    noise = rng.normal(0, 6, size=rgb.shape)
    rgb = np.clip(rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return rgb, mask, centerline, true_length


class SyntheticCenterlineDataset(Dataset):
    """In-memory synthetic dataset of single-shrimp centerline samples."""

    def __init__(
        self,
        length: int = 64,
        size: Tuple[int, int] = (128, 128),
        seed: int = 0,
        centerline_width: int = 1,
    ) -> None:
        self.length = length
        self.size = size
        self.seed = seed
        # Dilate the 1-px ground-truth centerline to a 3-px-wide band, matching
        # the paper's training protocol (Section 2.5.1) for stable gradients.
        self.centerline_width = centerline_width

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        rng = np.random.default_rng(self.seed + index)
        h, w = self.size
        rgb, mask, centerline, _ = generate_shrimp(h, w, rng)
        if self.centerline_width > 0:
            centerline = dilate_binary(centerline, iterations=self.centerline_width)
        image = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        return {
            "image": image,
            "mask": torch.from_numpy(mask).unsqueeze(0).float(),
            "target": torch.from_numpy(centerline).unsqueeze(0).float(),
        }


def materialize_dataset(
    root: str | Path,
    num_samples: int = 40,
    size: Tuple[int, int] = (128, 128),
    seed: int = 0,
) -> Path:
    """Write a synthetic centerline dataset to disk in the expected layout.

    Returns:
        The dataset root path.
    """
    root = Path(root)
    images_dir = root / "images"
    masks_dir = root / "masks"
    centerlines_dir = root / "centerlines"
    for directory in (images_dir, masks_dir, centerlines_dir):
        directory.mkdir(parents=True, exist_ok=True)

    h, w = size
    for i in range(num_samples):
        rng = np.random.default_rng(seed + i)
        rgb, mask, centerline, _ = generate_shrimp(h, w, rng)
        stem = f"shrimp_{i:04d}"
        Image.fromarray(rgb).save(images_dir / f"{stem}.png")
        Image.fromarray(mask * 255).save(masks_dir / f"{stem}.png")
        Image.fromarray(centerline * 255).save(centerlines_dir / f"{stem}.png")
    return root


def generate_scene(
    height: int = 256,
    width: int = 256,
    num_shrimp: int = 4,
    seed: int = 0,
) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray], List[float]]:
    """Generate a multi-shrimp scene for the full-framework demonstration.

    Returns:
        ``(rgb, masks, centerlines, true_lengths)``.
    """
    rng = np.random.default_rng(seed)
    background = rng.integers(150, 210, size=(height, width, 3), dtype=np.uint8)
    rgb = background.copy()
    masks: List[np.ndarray] = []
    centerlines: List[np.ndarray] = []
    lengths: List[float] = []
    for _ in range(num_shrimp):
        single_rgb, mask, centerline, length = generate_shrimp(height, width, rng)
        rgb[mask > 0] = single_rgb[mask > 0]
        masks.append(mask)
        centerlines.append(centerline)
        lengths.append(length)
    return rgb, masks, centerlines, lengths
