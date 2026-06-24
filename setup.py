"""Packaging for the dual-segmentation shrimp size-estimation framework."""

from pathlib import Path

from setuptools import find_packages, setup

_here = Path(__file__).parent
_long_description = (_here / "README.md").read_text(encoding="utf-8") if (_here / "README.md").exists() else ""

setup(
    name="shrimp-dualseg",
    version="1.0.0",
    description="Dual-Segmentation Framework for Automatic Detection and Size Estimation of Shrimp",
    long_description=_long_description,
    long_description_content_type="text/markdown",
    author="Reconstruction of Waqar et al., Sensors 2025 (25, 5830)",
    url="https://doi.org/10.3390/s25185830",
    license="MIT",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0",
        "torchvision>=0.15",
        "numpy>=1.24",
        "Pillow>=9.0",
        "PyYAML>=6.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0"],
        "onnx": ["onnx>=1.14", "onnxruntime>=1.16"],
        "coco": ["pycocotools>=2.0"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
)
