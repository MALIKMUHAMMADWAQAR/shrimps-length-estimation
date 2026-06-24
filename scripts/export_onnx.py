"""Export the centerline segmentation model to ONNX for deployment."""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import torch

from src.models import CenterlineSegmentationModel
from src.utils import load_checkpoint, load_config, setup_logger


class _CenterlineExportWrapper(torch.nn.Module):
    """Wrap the model so ONNX sees a single 4-channel (RGB+mask) input."""

    def __init__(self, model: CenterlineSegmentationModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, rgb_mask: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(rgb_mask[:, :3], rgb_mask[:, 3:4]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the centerline model to ONNX")
    parser.add_argument("--config", default="configs/infer.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default="checkpoints/centerline_model.onnx")
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--opset", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    logger = setup_logger("shrimp.export")

    model = CenterlineSegmentationModel(**cfg.model.get("centerline", {}))
    if args.checkpoint and Path(args.checkpoint).is_file():
        state = load_checkpoint(args.checkpoint)
        model.load_state_dict(state["model"])
        logger.info("Loaded checkpoint %s", args.checkpoint)
    model.eval()

    wrapper = _CenterlineExportWrapper(model).eval()
    dummy = torch.randn(1, 4, args.height, args.width)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    export_kwargs = dict(
        input_names=["rgb_mask"],
        output_names=["centerline"],
        opset_version=args.opset,
        dynamic_axes={"rgb_mask": {0: "batch", 2: "height", 3: "width"}, "centerline": {0: "batch"}},
    )
    try:
        # Use the stable TorchScript exporter, which avoids the optional
        # ``onnxscript`` dependency required by the newer dynamo exporter.
        torch.onnx.export(wrapper, dummy, args.output, dynamo=False, **export_kwargs)
    except TypeError:
        # Older torch versions do not accept the ``dynamo`` keyword.
        torch.onnx.export(wrapper, dummy, args.output, **export_kwargs)
    logger.info("Exported ONNX model to %s", args.output)


if __name__ == "__main__":
    main()
