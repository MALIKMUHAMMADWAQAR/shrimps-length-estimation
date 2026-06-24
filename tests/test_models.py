"""Forward-pass tests for the network components."""

from __future__ import annotations

import torch

from src.models import (
    ASPP,
    CenterlineSegmentationModel,
    CSPNeXt,
    CSPNeXtPAFPN,
    InputFusion,
    MobileNetV2Encoder,
    RTMDetIns,
)


def test_input_fusion_shape():
    fusion = InputFusion()
    image = torch.rand(2, 3, 64, 64)
    mask = torch.randint(0, 2, (2, 1, 64, 64)).float()
    out = fusion(image, mask)
    assert out.shape == (2, 3, 64, 64)


def test_mobilenet_encoder_strides():
    encoder = MobileNetV2Encoder()
    feats = encoder(torch.rand(1, 3, 128, 128))
    assert len(feats) == 5
    assert [f.shape[1] for f in feats] == list(encoder.out_channels)
    # Strides 2, 4, 8, 16, 32.
    assert feats[0].shape[-1] == 64
    assert feats[-1].shape[-1] == 4


def test_aspp_output_channels():
    aspp = ASPP(64, out_channels=32)
    out = aspp(torch.rand(1, 64, 16, 16))
    assert out.shape == (1, 32, 16, 16)


def test_centerline_model_forward_and_backward():
    model = CenterlineSegmentationModel()
    image = torch.rand(2, 3, 128, 128, requires_grad=True)
    mask = torch.randint(0, 2, (2, 1, 128, 128)).float()
    logits = model(image, mask)
    assert logits.shape == (2, 1, 128, 128)
    logits.sum().backward()
    assert image.grad is not None


def test_centerline_model_four_channel_input():
    model = CenterlineSegmentationModel()
    fused = torch.rand(1, 4, 96, 96)
    logits = model(fused)
    assert logits.shape == (1, 1, 96, 96)


def test_cspnext_backbone_levels():
    backbone = CSPNeXt()
    feats = backbone(torch.rand(1, 3, 256, 256))
    assert len(feats) == 3
    assert [f.shape[1] for f in feats] == backbone.out_channels


def test_pafpn_neck_uniform_channels():
    backbone = CSPNeXt()
    neck = CSPNeXtPAFPN(backbone.out_channels, out_channels=96)
    feats = neck(backbone(torch.rand(1, 3, 256, 256)))
    assert len(feats) == 3
    assert all(f.shape[1] == 96 for f in feats)


def test_rtmdet_forward_and_predict():
    model = RTMDetIns(num_classes=1).eval()
    x = torch.rand(1, 3, 256, 256)
    cls_scores, bbox_preds, kernel_preds, mask_feat = model(x)
    assert len(cls_scores) == 3
    assert mask_feat.shape[1] == model.mask_feat.num_prototypes
    results = model.predict(x, score_threshold=0.0, max_instances=5)
    assert len(results) == 1
    assert results[0].masks.shape[-2:] == (256, 256)
