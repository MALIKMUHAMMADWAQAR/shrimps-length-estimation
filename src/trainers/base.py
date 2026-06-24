"""Base trainer with AMP and (optional) distributed support."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from ..utils.distributed import is_main_process
from ..utils.misc import save_checkpoint


class BaseTrainer:
    """Common training scaffolding shared by the stage-specific trainers.

    Args:
        model: The network to optimise.
        optimizer: A configured optimizer.
        device: Target device.
        scheduler: Optional learning-rate scheduler stepped once per epoch.
        use_amp: Enable CUDA automatic mixed precision.
        grad_clip: Optional gradient-norm clipping value.
        logger: Optional logger.
        ckpt_dir: Directory where checkpoints are written.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        use_amp: bool = False,
        grad_clip: Optional[float] = None,
        logger: Optional[logging.Logger] = None,
        ckpt_dir: str | Path = "checkpoints",
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.use_amp = use_amp and device.type == "cuda"
        self.grad_clip = grad_clip
        self.logger = logger or logging.getLogger("shrimp")
        self.ckpt_dir = Path(ckpt_dir)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    def _optimizer_step(self, loss: torch.Tensor) -> None:
        """Backward + optimizer step with AMP scaling and optional clipping."""
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        if self.grad_clip is not None:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def save(self, name: str, **extra: object) -> Optional[Path]:
        """Persist a checkpoint (only on the main process)."""
        if not is_main_process():
            return None
        module = self.model.module if hasattr(self.model, "module") else self.model
        state = {"model": module.state_dict(), "optimizer": self.optimizer.state_dict(), **extra}
        path = self.ckpt_dir / name
        save_checkpoint(state, path)
        self.logger.info("Saved checkpoint to %s", path)
        return path
