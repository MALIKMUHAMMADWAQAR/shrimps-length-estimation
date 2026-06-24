"""Miscellaneous helpers: reproducibility, checkpoints and calibration."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """Count the (trainable) parameters of a model."""
    params = model.parameters()
    if trainable_only:
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def save_checkpoint(state: Dict[str, Any], path: str | Path) -> None:
    """Save a checkpoint dictionary, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str | Path, map_location: Optional[str] = "cpu") -> Dict[str, Any]:
    """Load a checkpoint dictionary."""
    return torch.load(path, map_location=map_location)


def compute_pixel_to_mm(reference_length_px: float, reference_length_mm: float) -> float:
    """Compute the pixel-to-millimetre calibration factor from a reference object.

    Args:
        reference_length_px: Measured length of the reference object in pixels.
        reference_length_mm: Known physical length of the reference object in mm.

    Returns:
        The millimetres-per-pixel conversion factor.
    """
    if reference_length_px <= 0:
        raise ValueError("reference_length_px must be positive")
    return reference_length_mm / reference_length_px
