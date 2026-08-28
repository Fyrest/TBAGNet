from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_split_integrity():
    manifest = json.loads((ROOT / "data/splits/railway_das/metadata.json").read_text(encoding="utf-8"))
    assert manifest["split_seed"] == 42
    assert manifest["sample_counts"] == {"train": 7936, "val": 928, "test": 2325}
    assert manifest["canonical_split_hashes"]["train"] == "8ba3f8e83c26deae6318fa784c9a93ffcb7521a2b5fd44fc1a339775a69e4888"
    assert manifest["canonical_split_hashes"]["val"] == "5720c7979742b10b12024162994e6d57b8f9d65373bf503317ce8d066b9f0758"
    assert manifest["canonical_split_hashes"]["test"] == "778f14c52e4e63aa5146205dfb84da72716fa8e8033c4d7b403b99d7c4c43938"
