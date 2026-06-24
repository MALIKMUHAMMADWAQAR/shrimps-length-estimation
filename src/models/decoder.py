"""U-Net style decoder for the centerline segmentation model.

The decoder progressively upsamples the ASPP context feature and fuses it with
the U-Net encoder skip connections (Section 2.5.1). Each fusion stage performs
convolutional refinement, after which a final convolution + sigmoid produces the
single-channel centerline mask.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from .fusion import FeatureFusion


class UNetDecoder(nn.Module):
    """Decoder that fuses ASPP context with encoder skips.

    Args:
        in_channels: Channels of the bottleneck (ASPP output) feature.
        skip_channels: Channels of each skip connection, ordered from the
            deepest (lowest resolution) to the shallowest (highest resolution).
        decoder_channels: Output channels produced after each fusion stage.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: Sequence[int],
        decoder_channels: Sequence[int],
    ) -> None:
        super().__init__()
        assert len(skip_channels) == len(decoder_channels), (
            "skip_channels and decoder_channels must have equal length"
        )
        self.blocks = nn.ModuleList()
        current = in_channels
        for skip_c, out_c in zip(skip_channels, decoder_channels):
            self.blocks.append(FeatureFusion(current, skip_c, out_c))
            current = out_c
        self.out_channels = decoder_channels[-1]

    def forward(self, x: torch.Tensor, skips: Sequence[torch.Tensor]) -> torch.Tensor:
        """Fuse the bottleneck feature with the ordered skip connections."""
        assert len(skips) == len(self.blocks), "Mismatch between skips and decoder blocks"
        for block, skip in zip(self.blocks, skips):
            x = block(x, skip)
        return x
