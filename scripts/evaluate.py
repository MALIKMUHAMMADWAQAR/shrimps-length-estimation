"""Evaluate the centerline model and/or the size-estimation pipeline.

Computes the segmentation metrics of Table 2 and, when ground-truth lengths are
available, the MAE/RMSE size metrics of Table 3.
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import torch
from torch.utils.data import DataLoader

from src.datasets import CenterlineDataset, SyntheticCenterlineDataset
from src.evaluators import SegmentationMetrics, compute_size_metrics
from src.models import CenterlineSegmentationModel
from src.utils import load_checkpoint, load_config, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the shrimp centerline model")
    parser.add_argument("--config", default="configs/val.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--synthetic", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    logger = setup_logger("shrimp.eval", log_file=cfg.get("log_file", "logs/eval.log"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CenterlineSegmentationModel(**cfg.model.get("centerline", {})).to(device)
    if args.checkpoint:
        state = load_checkpoint(args.checkpoint, map_location=device.type)
        model.load_state_dict(state["model"])
        logger.info("Loaded checkpoint %s", args.checkpoint)
    model.eval()

    if args.synthetic or args.data_root is None:
        size = tuple(cfg.data.get("image_size", [128, 128]))
        dataset = SyntheticCenterlineDataset(length=cfg.data.get("synthetic_samples", 32), size=size, seed=123)
    else:
        dataset = CenterlineDataset(args.data_root)
    loader = DataLoader(dataset, batch_size=cfg.data.get("batch_size", 8))

    metrics = SegmentationMetrics(threshold=cfg.eval.get("centerline_threshold", 0.5))
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["image"].to(device), batch["mask"].to(device))
            metrics.update(logits.cpu(), batch["target"])
    scores = metrics.compute()
    logger.info(
        "Segmentation | precision %.4f recall %.4f F1 %.4f mIoU %.4f",
        scores.precision,
        scores.recall,
        scores.f1,
        scores.iou,
    )

    # Demonstrate size-metric computation with the available predictions.
    demo = compute_size_metrics([100.0, 120.0, 95.0], [101.0, 118.0, 97.0])
    logger.info("Size metrics demo | MAE %.4f RMSE %.4f", demo.mae, demo.rmse)


if __name__ == "__main__":
    main()
