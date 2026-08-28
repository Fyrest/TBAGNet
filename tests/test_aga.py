from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for item in (str(SRC), str(ROOT)):
    if item not in sys.path:
        sys.path.insert(0, item)

import torch
from tbagnet.modules.aga import AddFusion, AdaptiveGatedAggregation, BRANCH_ORDER


def test_aga_softmax_across_branches():
    fusion = AdaptiveGatedAggregation(hidden_dim=128)
    features = {name: torch.randn(4, 128) for name in ["time", "fft", "cwt"]}
    fused, aux = fusion(features, ("time", "fft", "cwt"), return_weights=True)
    weights = aux["channel_weights"]
    assert tuple(fused.shape) == (4, 128)
    assert tuple(weights.shape) == (4, 3, 128)
    assert torch.allclose(weights.sum(dim=1), torch.ones(4, 128), atol=1e-6)


def test_add_fusion_is_sum_of_independently_normalized_branches():
    fusion = AddFusion(hidden_dim=8)
    features = {name: torch.randn(4, 8) for name in BRANCH_ORDER}

    fused, aux = fusion(features, BRANCH_ORDER, return_weights=True)
    expected = sum(fusion.norms[name](features[name]) for name in BRANCH_ORDER)

    assert fused.shape == (4, 8)
    assert torch.allclose(fused, expected)
    assert aux == {"enabled_views": BRANCH_ORDER, "fusion_type": "add"}
    assert not any("weight_mlp" in name or "gate_mlp" in name for name, _ in fusion.named_parameters())
