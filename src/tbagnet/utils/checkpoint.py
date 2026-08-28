from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def load_checkpoint_payload(path: str | Path, map_location="cpu") -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported checkpoint payload: {type(payload)}")
    return payload


def checkpoint_state_dict(payload: dict[str, Any]) -> dict[str, Any]:
    if "model_state" in payload:
        state_dict = payload["model_state"]
    elif "state_dict" in payload:
        state_dict = payload["state_dict"]
    elif "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
    else:
        state_dict = payload
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported state_dict payload: {type(state_dict)}")

    normalized: dict[str, Any] = {}
    for key, value in state_dict.items():
        normalized_key = key.removeprefix("module.")
        if normalized_key.startswith("channel_gate."):
            normalized_key = "aga." + normalized_key.removeprefix("channel_gate.")
        if normalized_key in normalized:
            raise KeyError(f"Checkpoint key normalization produced a duplicate key: {normalized_key}")
        normalized[normalized_key] = value
    return normalized


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
