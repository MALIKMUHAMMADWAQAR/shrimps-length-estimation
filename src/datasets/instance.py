"""COCO-style instance-segmentation dataset for the RTMDet stage.

A dependency-light reader that loads images, bounding boxes and polygon masks
from a COCO annotation file. ``pycocotools`` is only required for the official
AP evaluation (see :mod:`src.evaluators`), not for loading data here.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset


def polygons_to_mask(segmentation: List[List[float]], height: int, width: int) -> np.ndarray:
    """Rasterise COCO polygon segmentation into a binary mask."""
    mask = Image.new("L", (width, height), 0)
    drawer = ImageDraw.Draw(mask)
    for polygon in segmentation:
        if len(polygon) >= 6:
            drawer.polygon([tuple(polygon[i : i + 2]) for i in range(0, len(polygon), 2)], fill=1)
    return np.asarray(mask, dtype=np.uint8)


class CocoInstanceDataset(Dataset):
    """Instance-segmentation dataset reading COCO-format annotations.

    Args:
        image_dir: Directory containing the images.
        annotation_file: Path to the COCO JSON annotation file.
        transform: Optional transform applied to ``{"image": ...}``.
    """

    def __init__(
        self,
        image_dir: str | Path,
        annotation_file: str | Path,
        transform: Optional[Callable[[Dict], Dict]] = None,
    ) -> None:
        self.image_dir = Path(image_dir)
        with open(annotation_file, "r", encoding="utf-8") as handle:
            coco = json.load(handle)
        self.images = {img["id"]: img for img in coco["images"]}
        self.image_ids = list(self.images.keys())
        self.annotations: Dict[int, List[dict]] = defaultdict(list)
        for ann in coco.get("annotations", []):
            self.annotations[ann["image_id"]].append(ann)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        image_id = self.image_ids[index]
        info = self.images[image_id]
        path = self.image_dir / info["file_name"]
        image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        height, width = info["height"], info["width"]

        boxes: List[List[float]] = []
        labels: List[int] = []
        masks: List[np.ndarray] = []
        for ann in self.annotations[image_id]:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann.get("category_id", 1) - 1)
            seg = ann.get("segmentation", [])
            if isinstance(seg, list) and seg:
                masks.append(polygons_to_mask(seg, height, width))
            else:
                masks.append(np.zeros((height, width), dtype=np.uint8))

        sample = {
            "image": torch.from_numpy(image).permute(2, 0, 1).contiguous(),
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.long),
            "masks": torch.from_numpy(np.stack(masks)).float()
            if masks
            else torch.zeros((0, height, width)),
            "image_id": torch.tensor(image_id),
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


def instance_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, list]:
    """Collate variable-length instance annotations into lists."""
    collated: Dict[str, list] = {key: [item[key] for item in batch] for key in batch[0]}
    return collated
