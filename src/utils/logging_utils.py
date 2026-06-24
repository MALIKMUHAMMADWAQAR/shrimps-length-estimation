"""Logging helpers shared by the training / evaluation scripts."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


def setup_logger(name: str = "shrimp", log_file: Optional[str | Path] = None, level: int = logging.INFO) -> logging.Logger:
    """Create (or fetch) a configured logger.

    Args:
        name: Logger name.
        log_file: Optional path; when provided, logs are mirrored to this file.
        level: Logging level.

    Returns:
        The configured :class:`logging.Logger`.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    if logger.handlers:
        return logger

    formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger
