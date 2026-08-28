from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for item in (str(SRC), str(ROOT)):
    if item not in sys.path:
        sys.path.insert(0, item)

from tbagnet.model import TBAGNet
from tbagnet.utils.checkpoint import checkpoint_state_dict


def test_direct_model_defaults_are_paper_final():
    model = TBAGNet(num_classes=3)
    assert model.fusion_type == "aga"
    assert model.spectral_relation_depth == 4
    assert model.tfpatch_relation_depth == 1
    assert model.wavelet_conv_depth == 4
    assert model.pooling_mode == "max_only"
    assert model.cwt_branch.wavelet_mode == "real_morlet"


def test_checkpoint_metadata_is_not_used_as_model_configuration():
    expected = {"classifier.weight": torch.ones(2, 3)}
    payload = {
        "config": {"model": {"spectral_relation_depth": 999}},
        "run_id": "ignored",
        "model_state": expected,
    }

    extracted = checkpoint_state_dict(payload)

    assert extracted.keys() == expected.keys()
    assert torch.equal(extracted["classifier.weight"], expected["classifier.weight"])
