"""Lightweight image/mask transforms (no heavy third-party dependency)."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


class Resize:
    """Resize the image (bilinear) and the masks (nearest) to a fixed size."""

    def __init__(self, size: Tuple[int, int]) -> None:
        self.size = size

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        h, w = self.size
        sample["image"] = F.interpolate(
            sample["image"].unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False
        )[0]
        for key in ("mask", "target"):
            if key in sample:
                sample[key] = F.interpolate(
                    sample[key].unsqueeze(0).float(), size=(h, w), mode="nearest"
                )[0]
        return sample


class RandomHorizontalFlip:
    """Randomly flip the image and masks horizontally."""

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if torch.rand(1).item() < self.p:
            for key in ("image", "mask", "target"):
                if key in sample:
                    sample[key] = torch.flip(sample[key], dims=[-1])
        return sample


class Normalize:
    """Normalise the image with ImageNet-style mean / std."""

    def __init__(
        self,
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        sample["image"] = (sample["image"] - self.mean) / self.std
        return sample


class Compose:
    """Compose several transforms into a single callable."""

    def __init__(self, transforms: Sequence[object]) -> None:
        self.transforms = list(transforms)

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for transform in self.transforms:
            sample = transform(sample)  # type: ignore[operator]
        return sample


def dilate_binary(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Dilate a binary mask with a 3x3 structuring element (NumPy max filter)."""
    out = (np.asarray(mask) > 0).astype(np.uint8)
    for _ in range(iterations):
        padded = np.pad(out, 1, mode="constant")
        acc = np.zeros_like(out)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                acc |= padded[1 + dy : 1 + dy + out.shape[0], 1 + dx : 1 + dx + out.shape[1]]
        out = acc
    return out
