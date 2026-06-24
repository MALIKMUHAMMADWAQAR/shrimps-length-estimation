"""Reusable low-level neural network building blocks.

These helpers are shared across the RTMDet instance-segmentation backbone/neck/head
and the proposed centerline segmentation model.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

_ActLayer = Union[str, None]


def build_activation(activation: _ActLayer) -> nn.Module:
    """Create an activation module from a short string identifier.

    Args:
        activation: One of ``"relu"``, ``"relu6"``, ``"silu"``, ``"sigmoid"`` or ``None``.

    Returns:
        The instantiated activation module (``nn.Identity`` when ``None``).
    """
    if activation is None:
        return nn.Identity()
    activation = activation.lower()
    if activation == "relu":
        return nn.ReLU(inplace=True)
    if activation == "relu6":
        return nn.ReLU6(inplace=True)
    if activation == "silu":
        return nn.SiLU(inplace=True)
    if activation == "sigmoid":
        return nn.Sigmoid()
    if activation == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {activation!r}")


class ConvModule(nn.Module):
    """Convolution + (optional) normalization + (optional) activation.

    A light-weight re-implementation of the ``ConvModule`` used throughout
    OpenMMLab projects so that the package has no hard runtime dependency on
    ``mmcv``.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        stride: Convolution stride.
        padding: Explicit padding; if ``None`` it is inferred as ``kernel // 2 * dilation``.
        dilation: Convolution dilation.
        groups: Number of convolution groups (use ``in_channels`` for depthwise).
        bias: Whether to add a bias term; defaults to ``False`` when normalization is used.
        norm: Normalization type, one of ``"bn"`` or ``None``.
        activation: Activation identifier understood by :func:`build_activation`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Optional[int] = None,
        dilation: int = 1,
        groups: int = 1,
        bias: Optional[bool] = None,
        norm: Optional[str] = "bn",
        activation: _ActLayer = "silu",
    ) -> None:
        super().__init__()
        if padding is None:
            padding = (kernel_size // 2) * dilation
        if bias is None:
            bias = norm is None
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        if norm == "bn":
            self.norm: nn.Module = nn.BatchNorm2d(out_channels)
        elif norm is None:
            self.norm = nn.Identity()
        else:
            raise ValueError(f"Unsupported norm: {norm!r}")
        self.act = build_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution, normalization and activation in sequence."""
        return self.act(self.norm(self.conv(x)))


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution (depthwise 3x3 followed by pointwise 1x1)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        activation: _ActLayer = "silu",
    ) -> None:
        super().__init__()
        self.depthwise = ConvModule(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=in_channels,
            activation=activation,
        )
        self.pointwise = ConvModule(
            in_channels, out_channels, kernel_size=1, activation=activation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the depthwise then the pointwise convolution."""
        return self.pointwise(self.depthwise(x))


def make_divisible(value: float, divisor: int = 8, min_value: Optional[int] = None) -> int:
    """Round ``value`` to the nearest multiple of ``divisor`` (MobileNet style)."""
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)


def init_weights(module: nn.Module) -> None:
    """Kaiming-initialise convolutions and constant-initialise norm layers."""
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


def resize_like(x: torch.Tensor, reference: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    """Resize ``x`` so its spatial size matches ``reference``."""
    target_size: Tuple[int, int] = reference.shape[-2:]
    if x.shape[-2:] == target_size:
        return x
    align = False if mode in {"bilinear", "bicubic"} else None
    return nn.functional.interpolate(x, size=target_size, mode=mode, align_corners=align)
