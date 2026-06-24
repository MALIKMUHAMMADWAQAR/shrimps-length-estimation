"""Visualization helpers for qualitative results (Figures 6-7)."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw

_PALETTE: Sequence[Tuple[int, int, int]] = (
    (255, 99, 71),
    (60, 179, 113),
    (65, 105, 225),
    (255, 215, 0),
    (218, 112, 214),
    (0, 206, 209),
    (255, 140, 0),
    (147, 112, 219),
)


def _to_uint8_image(image: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert a CHW float tensor or HWC array into a HWC ``uint8`` array."""
    if isinstance(image, torch.Tensor):
        arr = image.detach().cpu().float()
        if arr.dim() == 3 and arr.shape[0] in (1, 3):
            arr = arr.permute(1, 2, 0)
        arr = arr.numpy()
    else:
        arr = np.asarray(image, dtype=np.float32)
    if arr.max() <= 1.0 + 1e-3:
        arr = arr * 255.0
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return np.clip(arr, 0, 255).astype(np.uint8)


def overlay_masks(
    image: torch.Tensor | np.ndarray,
    masks: Sequence[np.ndarray],
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend instance masks onto an image with per-instance colours."""
    canvas = _to_uint8_image(image).astype(np.float32)
    for idx, mask in enumerate(masks):
        colour = np.array(_PALETTE[idx % len(_PALETTE)], dtype=np.float32)
        binary = np.asarray(mask) > 0
        canvas[binary] = (1 - alpha) * canvas[binary] + alpha * colour
    return np.clip(canvas, 0, 255).astype(np.uint8)


def draw_centerlines_and_sizes(
    image: torch.Tensor | np.ndarray,
    centerlines: Sequence[np.ndarray],
    sizes_mm: Optional[Sequence[float]] = None,
    boxes: Optional[Sequence[Sequence[float]]] = None,
) -> Image.Image:
    """Render centerlines (and optional size labels) onto the image.

    Args:
        image: Background image.
        centerlines: One binary skeleton per shrimp.
        sizes_mm: Optional estimated sizes (mm) drawn as text labels.
        boxes: Optional xyxy boxes used to anchor the text labels.

    Returns:
        A PIL image with the overlaid annotations.
    """
    canvas = _to_uint8_image(image).copy()
    for idx, centerline in enumerate(centerlines):
        colour = _PALETTE[idx % len(_PALETTE)]
        canvas[np.asarray(centerline) > 0] = colour
    pil = Image.fromarray(canvas)
    drawer = ImageDraw.Draw(pil)
    if sizes_mm is not None:
        for idx, size in enumerate(sizes_mm):
            colour = _PALETTE[idx % len(_PALETTE)]
            if boxes is not None and idx < len(boxes):
                x, y = float(boxes[idx][0]), float(boxes[idx][1])
            else:
                ys, xs = np.where(np.asarray(centerlines[idx]) > 0)
                x, y = (float(xs.min()), float(ys.min())) if xs.size else (2.0, 2.0)
            drawer.text((max(x, 1), max(y - 10, 1)), f"{size:.1f} mm", fill=colour)
    return pil


def save_visualization(image: Image.Image | np.ndarray, path: str) -> None:
    """Save a visualization (PIL image or array) to disk."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image.save(path)
