"""CSPNeXtPAFPN neck for RTMDet.

A Path Aggregation Feature Pyramid Network (PAFPN) that performs a top-down then
a bottom-up pass over the CSPNeXt backbone features, aggregating multi-scale
context before the detection / mask heads (Section 2.4: "a BiFusion neck for
multi-scale feature aggregation").
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import CSPLayer
from .layers import ConvModule, init_weights


class CSPNeXtPAFPN(nn.Module):
    """Path-aggregation FPN built from CSP layers.

    Args:
        in_channels: Channels of the backbone outputs (P3, P4, P5).
        out_channels: Uniform channel count for every output level.
        num_csp_blocks: Number of CSP blocks per fusion layer.
        widen_factor: Width multiplier kept for API parity with RTMDet configs.
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int = 192,
        num_csp_blocks: int = 2,
        widen_factor: float = 0.75,
    ) -> None:
        super().__init__()
        self.in_channels = list(in_channels)
        self.out_channels = out_channels
        num_levels = len(in_channels)

        # Lateral 1x1 reductions for the top-down pathway.
        self.reduce_layers = nn.ModuleList(
            [ConvModule(c, out_channels, kernel_size=1, activation="silu") for c in in_channels]
        )
        self.top_down_blocks = nn.ModuleList(
            [
                CSPLayer(out_channels * 2, out_channels, num_blocks=num_csp_blocks, use_attention=False)
                for _ in range(num_levels - 1)
            ]
        )
        # Downsample + CSP layers for the bottom-up pathway.
        self.downsamples = nn.ModuleList(
            [
                ConvModule(out_channels, out_channels, kernel_size=3, stride=2, activation="silu")
                for _ in range(num_levels - 1)
            ]
        )
        self.bottom_up_blocks = nn.ModuleList(
            [
                CSPLayer(out_channels * 2, out_channels, num_blocks=num_csp_blocks, use_attention=False)
                for _ in range(num_levels - 1)
            ]
        )
        self.out_convs = nn.ModuleList(
            [ConvModule(out_channels, out_channels, kernel_size=3, activation="silu") for _ in range(num_levels)]
        )
        init_weights(self)

    def forward(self, features: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        """Run the top-down then bottom-up aggregation and return all levels."""
        assert len(features) == len(self.in_channels)
        # Lateral reduction.
        feats = [reduce_layer(f) for reduce_layer, f in zip(self.reduce_layers, features)]

        # Top-down pathway (from the coarsest level downward).
        inner_outs = [feats[-1]]
        for idx in range(len(feats) - 1, 0, -1):
            high = inner_outs[0]
            low = feats[idx - 1]
            upsampled = F.interpolate(high, size=low.shape[-2:], mode="nearest")
            fused = self.top_down_blocks[len(feats) - 1 - idx](torch.cat([upsampled, low], dim=1))
            inner_outs.insert(0, fused)

        # Bottom-up pathway (from the finest level upward).
        outs = [inner_outs[0]]
        for idx in range(len(feats) - 1):
            low = outs[-1]
            high = inner_outs[idx + 1]
            downsampled = self.downsamples[idx](low)
            fused = self.bottom_up_blocks[idx](torch.cat([downsampled, high], dim=1))
            outs.append(fused)

        return [out_conv(out) for out_conv, out in zip(self.out_convs, outs)]
