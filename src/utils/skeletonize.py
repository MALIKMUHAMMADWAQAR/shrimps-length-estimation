"""Zhang-Suen thinning (Zhang & Suen, 1984; Equation 9).

Converts a (few-pixel-wide) binary centerline mask into a one-pixel-wide
skeleton by iteratively removing boundary pixels while preserving connectivity.
Implemented in vectorised NumPy so it runs without SciPy / scikit-image.
"""

from __future__ import annotations

import numpy as np


def _neighbours_stack(padded: np.ndarray) -> list[np.ndarray]:
    """Return the 8 neighbours P2..P9 (clockwise from the top) of every pixel."""
    p2 = padded[:-2, 1:-1]
    p3 = padded[:-2, 2:]
    p4 = padded[1:-1, 2:]
    p5 = padded[2:, 2:]
    p6 = padded[2:, 1:-1]
    p7 = padded[2:, :-2]
    p8 = padded[1:-1, :-2]
    p9 = padded[:-2, :-2]
    return [p2, p3, p4, p5, p6, p7, p8, p9]


def _transitions(neigh: list[np.ndarray]) -> np.ndarray:
    """Count 0->1 transitions in the ordered neighbour sequence (A(p))."""
    seq = neigh + [neigh[0]]
    count = np.zeros_like(neigh[0], dtype=np.int32)
    for i in range(8):
        count += ((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.int32)
    return count


def zhang_suen_thinning(image: np.ndarray, max_iter: int = 100) -> np.ndarray:
    """Thin a binary image to a one-pixel-wide skeleton.

    Args:
        image: 2-D array; non-zero pixels are treated as foreground.
        max_iter: Safety cap on the number of thinning iterations.

    Returns:
        ``uint8`` array (same shape) with skeleton pixels set to 1.
    """
    img = (np.asarray(image) > 0).astype(np.uint8)
    if img.ndim != 2:
        raise ValueError("zhang_suen_thinning expects a 2-D array")

    for _ in range(max_iter):
        changed = False
        for step in range(2):
            padded = np.pad(img, 1, mode="constant", constant_values=0)
            neigh = _neighbours_stack(padded)
            p2, p3, p4, p5, p6, p7, p8, p9 = neigh
            b = sum(neigh)  # number of non-zero neighbours B(p)
            a = _transitions(neigh)  # 0->1 transitions A(p)

            cond_b = (b >= 2) & (b <= 6)
            cond_a = a == 1
            if step == 0:
                cond_1 = (p2 * p4 * p6) == 0
                cond_2 = (p4 * p6 * p8) == 0
            else:
                cond_1 = (p2 * p4 * p8) == 0
                cond_2 = (p2 * p6 * p8) == 0

            to_remove = (img == 1) & cond_b & cond_a & cond_1 & cond_2
            if to_remove.any():
                img[to_remove] = 0
                changed = True
        if not changed:
            break
    return img.astype(np.uint8)
