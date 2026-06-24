"""Train the centerline segmentation model (or smoke-train on synthetic data).

Examples
--------
Train on a real mask-to-centerline dataset::

    python scripts/train.py --config configs/train.yaml --data-root datasets/centerline

Quick end-to-end smoke run on procedurally generated shrimp::

    python scripts/train.py --config configs/train.yaml --synthetic --epochs 2
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401  (adds repo root to sys.path)
import torch
from torch.utils.data import DataLoader, random_split

from src.datasets import CenterlineDataset, SyntheticCenterlineDataset
from src.models import CenterlineSegmentationModel
from src.trainers import CenterlineTrainer
from src.utils import (
    cleanup_distributed,
    count_parameters,
    init_distributed,
    load_config,
    merge_overrides,
    set_seed,
    setup_logger,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the shrimp centerline model")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--data-root", default=None, help="Root of the mask-to-centerline dataset")
    parser.add_argument("--synthetic", action="store_true", help="Train on synthetic data")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision")
    parser.add_argument("opts", nargs=argparse.REMAINDER, help="key=value config overrides")
    return parser.parse_args()


def build_dataloaders(cfg, args):
    if args.synthetic or args.data_root is None:
        size = tuple(cfg.data.get("image_size", [128, 128]))
        full = SyntheticCenterlineDataset(length=cfg.data.get("synthetic_samples", 64), size=size)
        val_len = max(1, len(full) // 5)
        train_set, val_set = random_split(full, [len(full) - val_len, val_len])
    else:
        train_set = CenterlineDataset(f"{args.data_root}/train")
        val_set = CenterlineDataset(f"{args.data_root}/val")
    batch_size = args.batch_size or cfg.data.get("batch_size", 8)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=cfg.data.get("num_workers", 2))
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=cfg.data.get("num_workers", 2))
    return train_loader, val_loader


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.opts:
        cfg = merge_overrides(cfg, [o for o in args.opts if "=" in o])
    set_seed(cfg.get("seed", 42))
    logger = setup_logger("shrimp.train", log_file=cfg.get("log_file", "logs/train.log"))

    distributed, local_rank, device = init_distributed()
    logger.info("Device: %s | distributed: %s", device, distributed)

    model = CenterlineSegmentationModel(**cfg.model.get("centerline", {}))
    logger.info("Centerline model parameters: %.2fM", count_parameters(model) / 1e6)

    if distributed and device.type == "cuda":
        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    epochs = args.epochs or cfg.train.get("epochs", 40)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.train.get("lr", 0.02),
        momentum=cfg.train.get("momentum", 0.9),
        weight_decay=cfg.train.get("weight_decay", 1e-4),
    )
    milestones = cfg.train.get("lr_milestones", [int(epochs * 0.6), int(epochs * 0.85)])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    train_loader, val_loader = build_dataloaders(cfg, args)
    trainer = CenterlineTrainer(
        model,
        optimizer,
        device,
        scheduler=scheduler,
        use_amp=args.amp or cfg.train.get("amp", False),
        grad_clip=cfg.train.get("grad_clip", None),
        logger=logger,
        ckpt_dir=cfg.get("ckpt_dir", "checkpoints"),
    )
    history = trainer.fit(train_loader, val_loader, epochs=epochs)
    logger.info("Training complete: %s", history)
    cleanup_distributed()


if __name__ == "__main__":
    main()
