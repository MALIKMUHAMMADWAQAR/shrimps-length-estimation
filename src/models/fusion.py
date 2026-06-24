"""Fusion modules used by the centerline segmentation model.

Two kinds of fusion appear in the paper (Section 2.5.1, Equations 4-5):

* :class:`InputFusion` - early fusion of the RGB image with the binary instance
  mask. The 4-channel concatenation is reduced back to 3 channels with a 1x1
  convolution so that ImageNet-pretrained backbones remain compatible.
* :class:`FeatureFusion` - decoder-side fusion of an upsampled feature map with a
  U-Net skip connection, each fusion stage followed by convolutional refinement.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .layers import ConvModule, resize_like


class InputFusion(nn.Module):
    """Fuse RGB and a binary mask into a 3-channel tensor (Eqs. 4-5).

    The RGB image ``I`` (``H x W x 3``) is concatenated with the binary mask
    ``M`` (``H x W x 1``) to form a 4-channel tensor ``I_concat``; a 1x1
    convolution then projects it back to ``out_channels`` (3 by default):

        ``I_fused = Conv1x1(Concat(I, M))``.

    Args:
        in_channels: Channels of the RGB image (3).
        mask_channels: Channels of the mask (1).
        out_channels: Channels after fusion (3 to stay backbone compatible).
    """

    def __init__(self, in_channels: int = 3, mask_channels: int = 1, out_channels: int = 3) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.mask_channels = mask_channels
        self.out_channels = out_channels
        self.reduce = nn.Conv2d(in_channels + mask_channels, out_channels, kernel_size=1)

    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Concatenate ``image`` and ``mask`` along channels then 1x1-project.

        Args:
            image: RGB tensor of shape ``(N, 3, H, W)``.
            mask: Binary mask of shape ``(N, 1, H, W)``; resized to match ``image``.

        Returns:
            Fused tensor of shape ``(N, out_channels, H, W)``.
        """
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        mask = resize_like(mask, image, mode="nearest")
        concat = torch.cat([image, mask], dim=1)
        return self.reduce(concat)


class FeatureFusion(nn.Module):
    """Fuse an upsampled decoder feature with a U-Net skip connection.

    The upsampled feature is resized to the skip resolution, concatenated and
    refined with two convolutions (the "convolutional refinement layers"
    mentioned in Section 2.5.1).

    Args:
        in_channels: Channels of the (already upsampled) decoder feature.
        skip_channels: Channels of the encoder skip connection.
        out_channels: Output channels after refinement.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.refine = nn.Sequential(
            ConvModule(in_channels + skip_channels, out_channels, kernel_size=3, activation="relu"),
            ConvModule(out_channels, out_channels, kernel_size=3, activation="relu"),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Resize ``x`` to ``skip``, concatenate and refine."""
        x = resize_like(x, skip, mode="bilinear")
        return self.refine(torch.cat([x, skip], dim=1))
