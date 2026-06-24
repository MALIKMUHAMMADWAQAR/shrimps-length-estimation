"""Encoder backbones.

* :class:`MobileNetV2Encoder` - the lightweight encoder used by the proposed
  centerline segmentation model (Section 3.2.1: "all models utilized MobileNetV2
  as the encoder backbone").
* :class:`CSPNeXt` - the backbone of the RTMDet-ins-m instance-segmentation model
  (Section 2.4, Equation 1).

Both encoders are implemented from scratch so the package can run without
downloading external pretrained weights.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn

from .attention import ChannelAttention
from .layers import ConvModule, init_weights, make_divisible


# --------------------------------------------------------------------------- #
# MobileNetV2 encoder (centerline segmentation model)
# --------------------------------------------------------------------------- #
class InvertedResidual(nn.Module):
    """MobileNetV2 inverted residual block."""

    def __init__(self, in_channels: int, out_channels: int, stride: int, expand_ratio: int) -> None:
        super().__init__()
        assert stride in (1, 2)
        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            layers.append(ConvModule(in_channels, hidden_dim, kernel_size=1, activation="relu6"))
        layers.extend(
            [
                ConvModule(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=3,
                    stride=stride,
                    groups=hidden_dim,
                    activation="relu6",
                ),
                ConvModule(hidden_dim, out_channels, kernel_size=1, activation=None),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the inverted residual, adding the identity when shapes match."""
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2Encoder(nn.Module):
    """MobileNetV2 feature extractor returning 5 multi-scale feature maps.

    The encoder returns features at strides ``[2, 4, 8, 16, 32]`` with channel
    counts ``[16, 24, 32, 96, 320]`` (the standard MobileNetV2 taps used by
    segmentation libraries), which serve as U-Net skip connections.

    Args:
        in_channels: Number of input channels (3 for the fused image).
        width_mult: Width multiplier applied to all channels.
    """

    #: (expand_ratio, out_channels, num_blocks, stride) per inverted-residual stage.
    _SETTINGS: Sequence[Tuple[int, int, int, int]] = (
        (1, 16, 1, 1),
        (6, 24, 2, 2),
        (6, 32, 3, 2),
        (6, 64, 4, 2),
        (6, 96, 3, 1),
        (6, 160, 3, 2),
        (6, 320, 1, 1),
    )

    out_channels: Tuple[int, int, int, int, int]

    def __init__(self, in_channels: int = 3, width_mult: float = 1.0) -> None:
        super().__init__()
        input_channel = make_divisible(32 * width_mult)
        self.stem = ConvModule(in_channels, input_channel, kernel_size=3, stride=2, activation="relu6")

        # Build inverted residual stages, recording where the spatial stride changes
        # so we can emit feature maps at strides 4, 8, 16, 32.
        self.stages = nn.ModuleList()
        feature_channels: List[int] = [input_channel]  # stride 2 (stem output)
        current_channel = input_channel
        for expand_ratio, channel, num_blocks, stride in self._SETTINGS:
            output_channel = make_divisible(channel * width_mult)
            blocks: List[nn.Module] = []
            for block_idx in range(num_blocks):
                block_stride = stride if block_idx == 0 else 1
                blocks.append(
                    InvertedResidual(current_channel, output_channel, block_stride, expand_ratio)
                )
                current_channel = output_channel
            self.stages.append(nn.Sequential(*blocks))
            feature_channels.append(current_channel)

        # The taps used as skip connections (strides 2, 4, 8, 16, 32).
        self._return_stage_indices = (1, 2, 3, 5, 7)  # indices into the produced feature list
        self.out_channels = (16, 24, 32, 96, 320)
        init_weights(self)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Return feature maps at strides ``[2, 4, 8, 16, 32]``."""
        features: List[torch.Tensor] = []
        x = self.stem(x)
        features.append(x)  # stride 2
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return [features[i] for i in self._return_stage_indices]


# --------------------------------------------------------------------------- #
# CSPNeXt backbone (RTMDet-ins-m)
# --------------------------------------------------------------------------- #
class CSPNeXtBlock(nn.Module):
    """Bottleneck block of CSPNeXt using a 3x3 + 5x5 depthwise design."""

    def __init__(self, in_channels: int, out_channels: int, expansion: float = 0.5) -> None:
        super().__init__()
        hidden = int(out_channels * expansion)
        self.conv1 = ConvModule(in_channels, hidden, kernel_size=3, activation="silu")
        self.conv2 = ConvModule(hidden, out_channels, kernel_size=5, groups=hidden if hidden == out_channels else 1, activation="silu")
        self.add_identity = in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the two convolutions with an optional residual connection."""
        out = self.conv2(self.conv1(x))
        return x + out if self.add_identity else out


class CSPLayer(nn.Module):
    """Cross-stage partial layer with optional channel attention (RTMDet)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 1,
        expand_ratio: float = 0.5,
        use_attention: bool = True,
    ) -> None:
        super().__init__()
        mid_channels = int(out_channels * expand_ratio)
        self.main_conv = ConvModule(in_channels, mid_channels, kernel_size=1, activation="silu")
        self.short_conv = ConvModule(in_channels, mid_channels, kernel_size=1, activation="silu")
        self.blocks = nn.Sequential(
            *[CSPNeXtBlock(mid_channels, mid_channels) for _ in range(num_blocks)]
        )
        self.attention = ChannelAttention(2 * mid_channels) if use_attention else nn.Identity()
        self.final_conv = ConvModule(2 * mid_channels, out_channels, kernel_size=1, activation="silu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Split, process the main path, concatenate and fuse."""
        x_main = self.blocks(self.main_conv(x))
        x_short = self.short_conv(x)
        x_cat = torch.cat([x_main, x_short], dim=1)
        return self.final_conv(self.attention(x_cat))


class SPPFBottleneck(nn.Module):
    """Spatial pyramid pooling-fast layer used at the end of the backbone."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5) -> None:
        super().__init__()
        mid_channels = in_channels // 2
        self.conv1 = ConvModule(in_channels, mid_channels, kernel_size=1, activation="silu")
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.conv2 = ConvModule(mid_channels * 4, out_channels, kernel_size=1, activation="silu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool the feature map at multiple receptive fields and fuse."""
        x = self.conv1(x)
        p1 = self.pool(x)
        p2 = self.pool(p1)
        p3 = self.pool(p2)
        return self.conv2(torch.cat([x, p1, p2, p3], dim=1))


class CSPNeXt(nn.Module):
    """CSPNeXt backbone (RTMDet) emitting P3/P4/P5 feature maps.

    Args:
        in_channels: Number of input image channels (3).
        deepen_factor: Depth multiplier (0.67 for the ``-m`` variant).
        widen_factor: Width multiplier (0.75 for the ``-m`` variant).
        out_indices: Which stages to return (default the last three).
    """

    def __init__(
        self,
        in_channels: int = 3,
        deepen_factor: float = 0.67,
        widen_factor: float = 0.75,
        out_indices: Sequence[int] = (2, 3, 4),
    ) -> None:
        super().__init__()
        self.out_indices = tuple(out_indices)
        base_channels = [64, 128, 256, 512, 1024]
        base_depths = [3, 6, 6, 3]
        channels = [make_divisible(c * widen_factor) for c in base_channels]
        depths = [max(round(d * deepen_factor), 1) for d in base_depths]

        # Stem: three 3x3 convolutions (stride 2 overall).
        stem_channels = channels[0] // 2
        self.stem = nn.Sequential(
            ConvModule(in_channels, stem_channels, kernel_size=3, stride=2, activation="silu"),
            ConvModule(stem_channels, stem_channels, kernel_size=3, stride=1, activation="silu"),
            ConvModule(stem_channels, channels[0], kernel_size=3, stride=1, activation="silu"),
        )

        self.stages = nn.ModuleList()
        for i in range(4):
            in_c = channels[i]
            out_c = channels[i + 1]
            stage_layers: List[nn.Module] = [
                ConvModule(in_c, out_c, kernel_size=3, stride=2, activation="silu")
            ]
            if i == 3:
                stage_layers.append(SPPFBottleneck(out_c, out_c))
            stage_layers.append(CSPLayer(out_c, out_c, num_blocks=depths[i], use_attention=True))
            self.stages.append(nn.Sequential(*stage_layers))

        # Stage ``s`` (1-indexed in ``out_indices``) outputs ``channels[s]``.
        self.out_channels = [channels[i] for i in self.out_indices]
        init_weights(self)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Return the selected pyramid feature maps (default P3, P4, P5)."""
        outputs: List[torch.Tensor] = []
        x = self.stem(x)
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx + 1 in self.out_indices:
                outputs.append(x)
        return outputs
