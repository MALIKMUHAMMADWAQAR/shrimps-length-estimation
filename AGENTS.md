# AGENTS

Reference PyTorch reconstruction of *"A Dual-Segmentation Framework for the
Automatic Detection and Size Estimation of Shrimp"* (Waqar et al., **Sensors**
2025, 25(18), 5830, DOI `10.3390/s25185830`).

## Project layout

- `src/` — library code: `models/`, `datasets/`, `losses/`, `trainers/`,
  `evaluators/`, `utils/`, `visualization/`.
- `scripts/` — entry points: `train.py`, `evaluate.py`, `inference.py`, `export_onnx.py`.
- `configs/` — `train.yaml`, `val.yaml`, `infer.yaml`.
- `docs/` — structured paper summary (architecture / training / evaluation / dataset).
- `tests/` — unit + end-to-end tests.

## Cursor Cloud specific instructions

- **Package import path:** code is imported as the top-level `src` package
  (e.g. `from src.models import ...`). Scripts and tests add the repo root to
  `sys.path` automatically (`scripts/_bootstrap.py`, `tests/conftest.py`), so
  they run from the repo root without `pip install`. `pip install -e .` also works.
- **CPU vs GPU:** the cloud VM is CPU-only. The update script installs the CPU
  PyTorch wheels. The code is GPU/multi-GPU/AMP ready; `--amp` only activates on
  CUDA, and `torchrun --nproc_per_node=N scripts/train.py ...` enables DDP.
- **No dataset in repo:** the paper's datasets are private (available on request
  from the authors) and the paper PDF is not redistributed (see `paper/README.md`).
  Always develop/test with the built-in synthetic generator
  (`src/datasets/synthetic.py`): pass `--synthetic` to `train.py`/`evaluate.py`;
  `inference.py` generates a synthetic scene by default.
- **Run commands** (from repo root):
  - Tests: `python -m pytest -q`
  - Lint: `ruff check src scripts tests`
  - Train (smoke): `python scripts/train.py --config configs/train.yaml --synthetic --epochs 5`
  - Evaluate: `python scripts/evaluate.py --config configs/val.yaml --synthetic`
  - Inference: `python scripts/inference.py --checkpoint checkpoints/centerline_model.pth`
  - ONNX: `python scripts/export_onnx.py --checkpoint checkpoints/centerline_model.pth`
- **RTMDet detector is architecture-only.** `src/models/network.py::RTMDetIns`
  faithfully mirrors RTMDet-ins-m and exposes the losses, but it is **not** trained
  here (RTMDet's dynamic soft-label assignment is out of scope). Its
  `predict()`/`--full` path runs end-to-end but yields trivial masks until trained
  via the official OpenMMLab RTMDet recipe. The meaningful, trainable demo is the
  centerline model + Zhang-Suen skeletonization + size path (default `inference.py`).
- **ONNX export** uses the legacy TorchScript exporter (`dynamo=False`) so it does
  not need the optional `onnxscript` package.
- **Artifacts:** `checkpoints/`, `logs/`, `results/`, `datasets/` and model
  binaries (`*.pth`, `*.onnx`) are git-ignored; curated `assets/*.png` are tracked.
