from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for item in (str(SRC), str(ROOT)):
    if item not in sys.path:
        sys.path.insert(0, item)

import torch
from tbagnet.branches.wavelet import WaveletBranch


def test_real_morlet_shape_and_mode():
    branch = WaveletBranch(wavelet_mode="real_morlet", kernel_mean_subtraction=True, pooling_mode="max_only")
    x = torch.randn(2, 1, 1024)
    image = branch.transform(x)
    y = branch(x)
    assert branch.wavelet_mode == "real_morlet"
    assert branch.kernel_mean_subtraction is True
    assert tuple(image.shape) == (2, 1, 64, 64)
    assert tuple(y.shape) == (2, 128)
    assert torch.isfinite(image).all()
    assert torch.isfinite(y).all()
