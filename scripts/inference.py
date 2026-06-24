"""Run the dual-segmentation framework and visualise centerlines + sizes.

By default this generates a synthetic multi-shrimp scene, predicts the centerline
of every shrimp with the trained segmentation model, skeletonises it with the
Zhang-Suen algorithm and reports the estimated size (Figures 6-7). Pass
``--full`` to additionally run the RTMDet-ins detector to produce the instance
masks instead of using provided masks.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import torch

from src.datasets import generate_scene
from src.models import (
    CenterlineSegmentationModel,
    DualSegmentationFramework,
    FrameworkConfig,
    RTMDetIns,
)
from src.utils import (
    centerline_length_pixels,
    load_checkpoint,
    load_config,
    setup_logger,
    zhang_suen_thinning,
)
from src.visualization import draw_centerlines_and_sizes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run shrimp size-estimation inference")
    parser.add_argument("--config", default="configs/infer.yaml")
    parser.add_argument("--checkpoint", default=None, help="Centerline model checkpoint")
    parser.add_argument("--output", default="results/inference.png")
    parser.add_argument("--num-shrimp", type=int, default=4)
    parser.add_argument("--pixel-to-mm", type=float, default=0.5)
    parser.add_argument("--full", action="store_true", help="Use the RTMDet detector for masks")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    logger = setup_logger("shrimp.infer", log_file=cfg.get("log_file", "logs/infer.log"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    centerline_model = CenterlineSegmentationModel(**cfg.model.get("centerline", {})).to(device)
    if args.checkpoint and Path(args.checkpoint).is_file():
        state = load_checkpoint(args.checkpoint, map_location=device.type)
        centerline_model.load_state_dict(state["model"])
        logger.info("Loaded centerline checkpoint %s", args.checkpoint)
    centerline_model.eval()

    rgb, gt_masks, _, true_lengths = generate_scene(
        height=cfg.data.get("scene_size", 256),
        width=cfg.data.get("scene_size", 256),
        num_shrimp=args.num_shrimp,
        seed=args.seed,
    )
    image = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

    centerlines = []
    sizes_mm = []
    boxes = []

    if args.full:
        detector = RTMDetIns(num_classes=cfg.model.get("num_classes", 1)).to(device).eval()
        framework = DualSegmentationFramework(
            detector,
            centerline_model,
            FrameworkConfig(pixel_to_mm=args.pixel_to_mm),
        )
        results = framework(image)[0]
        logger.info("Detector produced %d instances", len(results))
        for measurement in results:
            centerlines.append(measurement.centerline.numpy())
            sizes_mm.append(measurement.length_mm)
            boxes.append(measurement.box.tolist())
    else:
        # Use the provided (ground-truth) instance masks; exercises the trained
        # centerline + skeletonisation + size-estimation path end-to-end.
        with torch.no_grad():
            for mask in gt_masks:
                mask_t = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0).to(device)
                logits = centerline_model(image, mask_t)
                prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
                centerline = (prob >= cfg.eval.get("centerline_threshold", 0.5)).astype(np.uint8)
                skeleton = zhang_suen_thinning(centerline)
                length_px = centerline_length_pixels(skeleton)
                centerlines.append(skeleton)
                sizes_mm.append(length_px * args.pixel_to_mm)
                ys, xs = np.where(mask > 0)
                boxes.append([xs.min(), ys.min(), xs.max(), ys.max()] if xs.size else [0, 0, 0, 0])

    for idx, (pred_mm, true_px) in enumerate(zip(sizes_mm, true_lengths)):
        logger.info(
            "Shrimp %d | predicted %.2f mm (true centerline %.1f px = %.2f mm)",
            idx,
            pred_mm,
            true_px,
            true_px * args.pixel_to_mm,
        )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    vis = draw_centerlines_and_sizes(rgb, centerlines, sizes_mm, boxes)
    vis.save(args.output)
    logger.info("Saved visualization to %s", args.output)


if __name__ == "__main__":
    main()
