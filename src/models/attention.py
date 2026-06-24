"""Attention modules.

``ChannelAttention`` reproduces the squeeze-and-excitation style attention used
inside the CSPNeXt blocks of RTMDet, while ``SpatialAttention`` / ``CBAM`` are
provided as optional refinement blocks for the decoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .layers import ConvModule


class ChannelAttention(nn.Module):
    """Squeeze-and-excitation channel attention (used by CSPNeXt).

    Args:
        channels: Number of input/output channels.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.act = nn.Hardsigmoid(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Re-weight channels of ``x`` with learned per-channel gates."""
        out = self.global_avgpool(x)
        out = self.fc(out)
        out = self.act(out)
        return x * out


class SpatialAttention(nn.Module):
    """Spatial attention sub-module (channel-pooled, 7x7 conv)."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Produce and apply a single-channel spatial attention map."""
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.act(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn


class CBAM(nn.Module):
    """Convolutional Block Attention Module (channel followed by spatial)."""

    def __init__(self, channels: int, spatial_kernel: int = 7) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(channels)
        self.spatial_attention = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel attention then spatial attention."""
        return self.spatial_attention(self.channel_attention(x))


class SqueezeExciteRefine(nn.Module):
    """Lightweight SE block with an explicit reduction ratio for the decoder."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.reduce = ConvModule(channels, hidden, kernel_size=1, norm=None, activation="relu")
        self.expand = ConvModule(hidden, channels, kernel_size=1, norm=None, activation="sigmoid")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Squeeze global context and excite channel responses."""
        scale = self.expand(self.reduce(self.pool(x)))
        return x * scale
