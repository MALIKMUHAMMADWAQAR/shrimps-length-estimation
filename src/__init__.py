"""Dual-Segmentation Framework for Automatic Detection and Size Estimation of Shrimp.

Reference implementation of:

    Waqar, M. M., Ali, H., Zhou, H., Mohamed, H. G., Kim, S. C., & Strzelecki, M. (2025).
    "A Dual-Segmentation Framework for the Automatic Detection and Size Estimation of Shrimp".
    Sensors, 25(18), 5830. https://doi.org/10.3390/s25185830

The package is organised into:

* :mod:`src.models`        - network definitions (RTMDet-ins, centerline model, framework).
* :mod:`src.losses`        - loss functions used to train both stages.
* :mod:`src.datasets`      - dataset readers for instance-segmentation and centerline data.
* :mod:`src.trainers`      - training loops (AMP / multi-GPU aware).
* :mod:`src.evaluators`    - segmentation / detection / size-estimation metrics.
* :mod:`src.utils`         - configuration, logging, skeletonization and calibration helpers.
* :mod:`src.visualization` - drawing utilities for qualitative results.
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
