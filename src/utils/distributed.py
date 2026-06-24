"""Helpers for (optional) multi-GPU distributed training.

The training scripts work transparently on CPU, a single GPU, or multiple GPUs
launched with ``torchrun``. These helpers centralise the environment handling.
"""

from __future__ import annotations

import os
from typing import Tuple

import torch
import torch.distributed as dist


def is_distributed() -> bool:
    """Return ``True`` when running under a distributed launcher."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Return the global rank (0 when not distributed)."""
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    """Return the total number of processes (1 when not distributed)."""
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    """Return ``True`` on the rank-0 process."""
    return get_rank() == 0


def init_distributed() -> Tuple[bool, int, torch.device]:
    """Initialise the process group from environment variables if present.

    Returns:
        A tuple ``(distributed, local_rank, device)``.
    """
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return False, 0, device

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return True, local_rank, device


def cleanup_distributed() -> None:
    """Destroy the process group if it was initialised."""
    if is_distributed():
        dist.destroy_process_group()
