"""Mask-to-centerline dataset for the semantic-segmentation stage.

Expected directory layout (Section 2.2, second dataset)::

    root/
      images/      <stem>.png   # RGB crop containing a single shrimp
      masks/       <stem>.png   # binary instance mask of the shrimp
      centerlines/ <stem>.png   # binary ground-truth centerline

The ground-truth centerline is dilated to a three-pixel-wide band at load time,
matching the paper's training protocol.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .transforms import dilate_binary


def _list_stems(directory: Path) -> List[str]:
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    return sorted(p.stem for p in directory.iterdir() if p.suffix.lower() in exts)


class CenterlineDataset(Dataset):
    """Dataset yielding ``(image, mask, target)`` triplets for centerline prediction.

    Args:
        root: Dataset root containing ``images``, ``masks`` and ``centerlines``.
        transform: Optional transform applied to the assembled sample dict.
        centerline_width: Dilation iterations to reach the 3-pixel-wide target.
    """

    def __init__(
        self,
        root: str | Path,
        transform: Optional[Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]] = None,
        centerline_width: int = 1,
    ) -> None:
        self.root = Path(root)
        self.images_dir = self.root / "images"
        self.masks_dir = self.root / "masks"
        self.centerlines_dir = self.root / "centerlines"
        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        self.stems = _list_stems(self.images_dir)
        self.transform = transform
        self.centerline_width = centerline_width

    def __len__(self) -> int:
        return len(self.stems)

    def _load_image(self, path: Path) -> torch.Tensor:
        arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    def _load_binary(self, path: Path) -> np.ndarray:
        return (np.asarray(Image.open(path).convert("L")) > 127).astype(np.uint8)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        stem = self.stems[index]
        image = self._load_image(self.images_dir / f"{stem}.png")
        mask = self._load_binary(self.masks_dir / f"{stem}.png")
        centerline = self._load_binary(self.centerlines_dir / f"{stem}.png")
        if self.centerline_width > 0:
            centerline = dilate_binary(centerline, iterations=self.centerline_width)

        sample: Dict[str, torch.Tensor] = {
            "image": image,
            "mask": torch.from_numpy(mask).unsqueeze(0).float(),
            "target": torch.from_numpy(centerline).unsqueeze(0).float(),
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
