from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for item in (str(SRC), str(ROOT)):
    if item not in sys.path:
        sys.path.insert(0, item)

import torch
from tbagnet.branches.spectral import SpectralBranch
from tbagnet.branches.wavelet import TimeFrequencyPatchBranch


def test_patch_aggregation_modes():
    x = torch.randn(2, 1, 1024)
    image = torch.randn(2, 1, 64, 64)
    expected = {
        "avg_only": (True, False),
        "max_only": (False, True),
        "dual_pool": (True, True),
    }
    for mode, flags in expected.items():
        spectral = SpectralBranch(spectral_relation_depth=4, pooling_mode=mode)
        tfpatch = TimeFrequencyPatchBranch(tfpatch_relation_depth=1, pooling_mode=mode)
        sf = spectral(x)
        tf = tfpatch(image)
        assert tuple(spectral.last_token_shape) == (2, 64, 128)
        assert tuple(tfpatch.last_token_shape) == (2, 49, 128)
        assert tuple(sf.shape) == (2, 128)
        assert tuple(tf.shape) == (2, 128)
        assert (spectral.last_used_avg_pool, spectral.last_used_max_pool) == flags
        assert (tfpatch.last_used_avg_pool, tfpatch.last_used_max_pool) == flags
