from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def split_hash(csv_path: str | Path) -> str:
    df = pd.read_csv(csv_path)
    rows = df.astype(str).agg("|".join, axis=1).tolist()
    return hashlib.sha256(("\n".join(rows)).encode("utf-8")).hexdigest()


def load_split_manifest(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
