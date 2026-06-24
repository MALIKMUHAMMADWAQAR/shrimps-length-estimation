"""Atrous Spatial Pyramid Pooling (ASPP).

Implements the ASPP module of DeepLabv3 (Chen et al., ECCV 2018), which is one of
the two parallel branches of the proposed centerline segmentation model. ASPP
captures multi-scale context through parallel atrous convolutions with different
dilation rates plus an image-level global pooling branch.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import ConvModule


class ASPPPooling(nn.Module):
    """Image-level (global average pooling) branch of ASPP."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        # No BatchNorm here: the pooled feature is 1x1, so BN would fail for a
        # batch size of 1 and is unnecessary after global pooling.
        self.conv = ConvModule(in_channels, out_channels, kernel_size=1, norm=None, activation="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool to a single vector and broadcast back to the input resolution."""
        size = x.shape[-2:]
        out = self.conv(self.gap(x))
        return F.interpolate(out, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels for the projected fused feature.
        atrous_rates: Dilation rates for the parallel atrous convolutions.
        dropout: Dropout probability applied to the projected output.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 256,
        atrous_rates: Sequence[int] = (6, 12, 18),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        branches = [ConvModule(in_channels, out_channels, kernel_size=1, activation="relu")]
        for rate in atrous_rates:
            branches.append(
                ConvModule(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=rate,
                    dilation=rate,
                    activation="relu",
                )
            )
        branches.append(ASPPPooling(in_channels, out_channels))
        self.branches = nn.ModuleList(branches)
        self.project = nn.Sequential(
            ConvModule(out_channels * len(branches), out_channels, kernel_size=1, activation="relu"),
            nn.Dropout2d(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenate all pyramid branches and project to ``out_channels``."""
        features = [branch(x) for branch in self.branches]
        return self.project(torch.cat(features, dim=1))
