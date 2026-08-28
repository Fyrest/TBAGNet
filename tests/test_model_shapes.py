from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for item in (str(SRC), str(ROOT)):
    if item not in sys.path:
        sys.path.insert(0, item)

import torch
import yaml
from tbagnet.model import build_model


def test_model_shapes_cpu():
    cfg = yaml.safe_load((ROOT / "configs/railway_das.yaml").read_text(encoding="utf-8"))
    model = build_model(cfg).cpu()
    x = torch.randn(2, 1, 1024)
    out = model(x, return_features=True)
    assert tuple(out["logits"].shape) == (2, 6)
    assert tuple(out["features"]["time"].shape) == (2, 128)
    assert tuple(out["features"]["fft"].shape) == (2, 128)
    assert tuple(out["features"]["cwt"].shape) == (2, 128)
