"""Size-estimation metrics: Mean Absolute Error and Root Mean Squared Error.

Implements Equations 10-11 (Table 3). Errors can be reported in pixels and in
millimetres by supplying the appropriate predicted / target sequences.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class SizeScores:
    """MAE / RMSE pair for size estimation."""

    mae: float
    rmse: float


def compute_size_metrics(predictions: Sequence[float], targets: Sequence[float]) -> SizeScores:
    """Compute MAE and RMSE between predicted and ground-truth lengths.

    Args:
        predictions: Predicted lengths.
        targets: Ground-truth lengths (same unit and order as ``predictions``).

    Returns:
        A :class:`SizeScores` with the MAE and RMSE.
    """
    pred = np.asarray(predictions, dtype=np.float64)
    tgt = np.asarray(targets, dtype=np.float64)
    if pred.shape != tgt.shape:
        raise ValueError("predictions and targets must have the same shape")
    if pred.size == 0:
        return SizeScores(mae=0.0, rmse=0.0)
    errors = pred - tgt
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    return SizeScores(mae=mae, rmse=rmse)
