"""Top-level network definitions for the dual-segmentation framework.

This module assembles the building blocks into three public models:

* :class:`CenterlineSegmentationModel` - the paper's proposed enhanced
  segmentation model (DeepLabv3 ASPP context + U-Net feature fusion) that
  predicts the shrimp centerline from a fused RGB+mask input.
* :class:`RTMDetIns` - a from-scratch RTMDet-ins-m instance segmentation model
  (CSPNeXt backbone + CSPNeXtPAFPN neck + decoupled dynamic-mask head).
* :class:`DualSegmentationFramework` - the full pipeline that chains instance
  segmentation, per-instance centerline prediction and size estimation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aspp import ASPP
from .decoder import UNetDecoder
from .encoder import CSPNeXt, MobileNetV2Encoder
from .fusion import InputFusion
from .head import MaskFeatModule, RTMDetInsHead, SegmentationHead
from .layers import resize_like
from .neck import CSPNeXtPAFPN


# --------------------------------------------------------------------------- #
# Proposed centerline segmentation model
# --------------------------------------------------------------------------- #
class CenterlineSegmentationModel(nn.Module):
    """Enhanced DeepLabv3 + U-Net model for shrimp centerline prediction.

    Implements Section 2.5.1 / Equations 4-6. A fused RGB+mask image is fed in
    parallel to a MobileNetV2 (U-Net) encoder and an ASPP context module; the
    ASPP output is decoded with U-Net skip fusion to yield the centerline.

    Args:
        in_channels: RGB channels (3).
        mask_channels: Binary mask channels (1).
        num_classes: Output channels (1 for binary centerline).
        aspp_channels: ASPP output channel count.
        atrous_rates: Dilation rates used inside ASPP.
        decoder_channels: Output channels after each decoder fusion stage.
        width_mult: MobileNetV2 width multiplier.
    """

    def __init__(
        self,
        in_channels: int = 3,
        mask_channels: int = 1,
        num_classes: int = 1,
        aspp_channels: int = 256,
        atrous_rates: Sequence[int] = (6, 12, 18),
        decoder_channels: Sequence[int] = (128, 64, 48, 32),
        width_mult: float = 1.0,
    ) -> None:
        super().__init__()
        self.input_fusion = InputFusion(in_channels, mask_channels, out_channels=3)
        self.encoder = MobileNetV2Encoder(in_channels=3, width_mult=width_mult)
        enc_channels = self.encoder.out_channels  # (16, 24, 32, 96, 320)

        # ASPP is guided by the deepest encoder feature concatenated with the
        # downsampled fused image (extra spatial cues, Section 2.5.1).
        self.aspp = ASPP(enc_channels[-1] + 3, out_channels=aspp_channels, atrous_rates=atrous_rates)

        # Decoder fuses from the deepest skip to the shallowest.
        skip_channels = list(reversed(enc_channels[:-1]))  # (96, 32, 24, 16)
        self.decoder = UNetDecoder(aspp_channels, skip_channels, decoder_channels)
        self.seg_head = SegmentationHead(decoder_channels[-1], num_classes, upsample_factor=2)

    def forward(self, image: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Predict centerline logits for a fused RGB+mask input.

        Args:
            image: RGB tensor ``(N, 3, H, W)``. If ``mask`` is ``None``, ``image``
                is assumed to already carry the mask in a 4th channel.
            mask: Optional binary mask ``(N, 1, H, W)``.

        Returns:
            Centerline logits of shape ``(N, num_classes, H, W)``.
        """
        if mask is None:
            assert image.shape[1] == 4, "Expected a 4-channel input when mask is None"
            image, mask = image[:, :3], image[:, 3:4]
        fused = self.input_fusion(image, mask)

        skips = self.encoder(fused)  # strides [2, 4, 8, 16, 32]
        deepest = skips[-1]
        guide = resize_like(fused, deepest, mode="bilinear")
        context = self.aspp(torch.cat([deepest, guide], dim=1))

        decoder_skips = list(reversed(skips[:-1]))  # [s16, s8, s4, s2]
        decoded = self.decoder(context, decoder_skips)
        logits = self.seg_head(decoded)
        return resize_like(logits, image, mode="bilinear")


# --------------------------------------------------------------------------- #
# RTMDet-ins instance segmentation model
# --------------------------------------------------------------------------- #
def _parse_dynamic_params(
    flat_params: torch.Tensor, weight_nums: Sequence[int], bias_nums: Sequence[int]
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Split a flat parameter vector into per-layer dynamic conv weights/biases."""
    num_instances = flat_params.size(0)
    num_layers = len(weight_nums)
    params_splits = list(torch.split_with_sizes(flat_params, list(weight_nums) + list(bias_nums), dim=1))
    weight_splits = params_splits[:num_layers]
    bias_splits = params_splits[num_layers:]
    in_dims: List[int] = []
    out_dims: List[int] = []
    running_in = None
    for i, wn in enumerate(weight_nums):
        out_dim = bias_nums[i]
        in_dim = wn // out_dim
        in_dims.append(in_dim)
        out_dims.append(out_dim)
        running_in = in_dim
    _ = running_in
    weights = [
        w.reshape(num_instances * out_dims[i], in_dims[i], 1, 1)
        for i, w in enumerate(weight_splits)
    ]
    biases = [b.reshape(num_instances * out_dims[i]) for i, b in enumerate(bias_splits)]
    return weights, biases


@dataclass
class InstanceSegOutput:
    """Container for instance-segmentation predictions of a single image."""

    boxes: torch.Tensor  # (K, 4) xyxy
    scores: torch.Tensor  # (K,)
    labels: torch.Tensor  # (K,)
    masks: torch.Tensor  # (K, H, W) float probabilities


class RTMDetIns(nn.Module):
    """RTMDet-ins-m instance segmentation network (Section 2.4).

    Args:
        num_classes: Number of classes (1 = shrimp).
        deepen_factor: Backbone depth multiplier (0.67 for ``-m``).
        widen_factor: Backbone width multiplier (0.75 for ``-m``).
        neck_channels: Channel count of every neck output level.
        num_prototypes: Number of mask prototype channels.
    """

    def __init__(
        self,
        num_classes: int = 1,
        deepen_factor: float = 0.67,
        widen_factor: float = 0.75,
        neck_channels: int = 192,
        num_prototypes: int = 8,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.backbone = CSPNeXt(deepen_factor=deepen_factor, widen_factor=widen_factor)
        self.neck = CSPNeXtPAFPN(self.backbone.out_channels, out_channels=neck_channels, widen_factor=widen_factor)
        self.head = RTMDetInsHead(
            neck_channels,
            num_classes=num_classes,
            feat_channels=neck_channels,
            num_prototypes=num_prototypes,
            num_levels=len(self.backbone.out_channels),
        )
        self.mask_feat = MaskFeatModule(neck_channels, num_prototypes=num_prototypes)
        # Pyramid strides for P3, P4, P5.
        self.strides: Tuple[int, ...] = (8, 16, 32)
        self.mask_stride = 8

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        """Return raw head outputs and the mask prototype features."""
        backbone_feats = self.backbone(x)
        neck_feats = self.neck(backbone_feats)
        cls_scores, bbox_preds, kernel_preds = self.head(neck_feats)
        mask_features = self.mask_feat(neck_feats[0])
        return cls_scores, bbox_preds, kernel_preds, mask_features

    @staticmethod
    def _grid_points(height: int, width: int, stride: int, device: torch.device) -> torch.Tensor:
        """Return the centre coordinate (x, y) of each location at a level."""
        shift_x = (torch.arange(width, device=device) + 0.5) * stride
        shift_y = (torch.arange(height, device=device) + 0.5) * stride
        yy, xx = torch.meshgrid(shift_y, shift_x, indexing="ij")
        return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        score_threshold: float = 0.3,
        max_instances: int = 100,
    ) -> List[InstanceSegOutput]:
        """Decode instance masks from a forward pass (inference only).

        This performs a lightweight decode (per-location scoring, top-k
        selection and dynamic-mask generation) sufficient to run the full
        pipeline end-to-end. Training of the detector follows the standard
        RTMDet dynamic-label-assignment recipe.
        """
        cls_scores, bbox_preds, kernel_preds, mask_features = self.forward(x)
        batch_size = x.size(0)
        img_h, img_w = x.shape[-2:]
        device = x.device
        results: List[InstanceSegOutput] = []

        proto_h, proto_w = mask_features.shape[-2:]
        for b in range(batch_size):
            all_scores: List[torch.Tensor] = []
            all_boxes: List[torch.Tensor] = []
            all_kernels: List[torch.Tensor] = []
            all_centers: List[torch.Tensor] = []
            for level, (cls, box, kernel) in enumerate(zip(cls_scores, bbox_preds, kernel_preds)):
                stride = self.strides[level]
                h, w = cls.shape[-2:]
                points = self._grid_points(h, w, stride, device)  # (HW, 2)
                cls_b = cls[b].permute(1, 2, 0).reshape(-1, self.num_classes).sigmoid()
                box_b = box[b].permute(1, 2, 0).reshape(-1, 4) * stride
                kernel_b = kernel[b].permute(1, 2, 0).reshape(points.size(0), -1)
                score, _ = cls_b.max(dim=1)
                # distance (l, t, r, b) -> xyxy.
                x1 = points[:, 0] - box_b[:, 0]
                y1 = points[:, 1] - box_b[:, 1]
                x2 = points[:, 0] + box_b[:, 2]
                y2 = points[:, 1] + box_b[:, 3]
                boxes = torch.stack([x1, y1, x2, y2], dim=1)
                all_scores.append(score)
                all_boxes.append(boxes)
                all_kernels.append(kernel_b)
                all_centers.append(points)

            scores = torch.cat(all_scores)
            boxes = torch.cat(all_boxes)
            kernels = torch.cat(all_kernels)
            centers = torch.cat(all_centers)

            keep = scores >= score_threshold
            if keep.sum() == 0:
                # Fall back to the single most confident location.
                keep = scores >= scores.max()
            scores, boxes, kernels, centers = scores[keep], boxes[keep], kernels[keep], centers[keep]

            topk = min(max_instances, scores.numel())
            order = scores.argsort(descending=True)[:topk]
            scores, boxes, kernels, centers = scores[order], boxes[order], kernels[order], centers[order]

            masks = self._dynamic_masks(mask_features[b], kernels, centers, proto_h, proto_w)
            masks = F.interpolate(masks.unsqueeze(0), size=(img_h, img_w), mode="bilinear", align_corners=False)[0]
            labels = torch.zeros(scores.numel(), dtype=torch.long, device=device)
            results.append(InstanceSegOutput(boxes=boxes, scores=scores, labels=labels, masks=masks.sigmoid()))
        return results

    def _dynamic_masks(
        self,
        prototypes: torch.Tensor,
        kernels: torch.Tensor,
        centers: torch.Tensor,
        proto_h: int,
        proto_w: int,
    ) -> torch.Tensor:
        """Apply per-instance dynamic convolutions to the prototype features."""
        num_inst = kernels.size(0)
        if num_inst == 0:
            return prototypes.new_zeros((0, proto_h, proto_w))

        device = prototypes.device
        ys = torch.linspace(0, proto_h - 1, proto_h, device=device)
        xs = torch.linspace(0, proto_w - 1, proto_w, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        grid_x = grid_x[None].expand(num_inst, -1, -1)
        grid_y = grid_y[None].expand(num_inst, -1, -1)
        cx = (centers[:, 0] / self.mask_stride).clamp(0, proto_w - 1)[:, None, None]
        cy = (centers[:, 1] / self.mask_stride).clamp(0, proto_h - 1)[:, None, None]
        rel_x = (grid_x - cx) / max(proto_w, 1)
        rel_y = (grid_y - cy) / max(proto_h, 1)
        coord = torch.stack([rel_x, rel_y], dim=1)  # (N, 2, H, W)

        proto = prototypes[None].expand(num_inst, -1, -1, -1)
        feat = torch.cat([proto, coord], dim=1)  # (N, P+2, H, W)
        n, c, h, w = feat.shape
        feat = feat.reshape(1, n * c, h, w)

        weights, biases = _parse_dynamic_params(kernels, self.head.weight_nums, self.head.bias_nums)
        for i, (weight, bias) in enumerate(zip(weights, biases)):
            feat = F.conv2d(feat, weight, bias=bias, stride=1, padding=0, groups=num_inst)
            if i < len(weights) - 1:
                feat = F.relu(feat)
        return feat.reshape(num_inst, h, w)


# --------------------------------------------------------------------------- #
# Full dual-segmentation framework
# --------------------------------------------------------------------------- #
@dataclass
class ShrimpMeasurement:
    """Per-shrimp measurement produced by the framework."""

    box: torch.Tensor
    instance_mask: torch.Tensor
    centerline: torch.Tensor  # one-pixel-wide skeleton (H, W) uint8
    length_px: float
    length_mm: float


@dataclass
class FrameworkConfig:
    """Configuration for :class:`DualSegmentationFramework`."""

    score_threshold: float = 0.3
    mask_threshold: float = 0.5
    centerline_threshold: float = 0.5
    pixel_to_mm: float = 1.0
    max_instances: int = 100


class DualSegmentationFramework(nn.Module):
    """End-to-end shrimp detection, centerline prediction and size estimation.

    The framework chains the two trained models (Section 2.3): instance
    segmentation isolates individual shrimp, the centerline model predicts each
    shrimp's medial line, and a Zhang-Suen thinning + arc-length conversion
    yields the size in millimetres.

    Args:
        detector: An :class:`RTMDetIns` (or compatible) instance segmenter.
        centerline_model: A :class:`CenterlineSegmentationModel`.
        config: Inference thresholds and the pixel-to-mm calibration factor.
    """

    def __init__(
        self,
        detector: nn.Module,
        centerline_model: CenterlineSegmentationModel,
        config: Optional[FrameworkConfig] = None,
    ) -> None:
        super().__init__()
        self.detector = detector
        self.centerline_model = centerline_model
        self.config = config or FrameworkConfig()

    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> List[List[ShrimpMeasurement]]:
        """Run the full pipeline on a batch of RGB images.

        Args:
            image: RGB tensor ``(N, 3, H, W)`` in ``[0, 1]``.

        Returns:
            For each image, a list of :class:`ShrimpMeasurement` objects.
        """
        # Local import avoids a hard dependency cycle with the utils package.
        from ..utils.skeletonize import zhang_suen_thinning
        from ..utils.geometry import centerline_length_pixels

        instances = self.detector.predict(
            image,
            score_threshold=self.config.score_threshold,
            max_instances=self.config.max_instances,
        )
        batch_results: List[List[ShrimpMeasurement]] = []
        for b, inst in enumerate(instances):
            measurements: List[ShrimpMeasurement] = []
            if inst.masks.numel() == 0:
                batch_results.append(measurements)
                continue
            bin_masks = (inst.masks >= self.config.mask_threshold).float()
            rgb = image[b : b + 1]
            for k in range(bin_masks.size(0)):
                mask = bin_masks[k : k + 1].unsqueeze(0)  # (1,1,H,W)
                logits = self.centerline_model(rgb, mask)
                prob = torch.sigmoid(logits)[0, 0]
                centerline = (prob >= self.config.centerline_threshold).to(torch.uint8)
                skeleton = zhang_suen_thinning(centerline.cpu().numpy())
                length_px = centerline_length_pixels(skeleton)
                length_mm = length_px * self.config.pixel_to_mm
                measurements.append(
                    ShrimpMeasurement(
                        box=inst.boxes[k].cpu(),
                        instance_mask=bin_masks[k].cpu(),
                        centerline=torch.from_numpy(skeleton),
                        length_px=length_px,
                        length_mm=length_mm,
                    )
                )
            batch_results.append(measurements)
        return batch_results
