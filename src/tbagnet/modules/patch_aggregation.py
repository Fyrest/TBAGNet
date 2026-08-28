from __future__ import annotations

import torch

POOLING_MODES = {"avg_only", "max_only", "dual_pool"}


def normalize_pooling_mode(pooling_mode: str) -> str:
    mode = str(pooling_mode).lower()
    if mode not in POOLING_MODES:
        raise ValueError("pooling_mode must be avg_only, max_only, or dual_pool.")
    return mode


def aggregate_patch_tokens(tokens: torch.Tensor, pooling_mode: str) -> torch.Tensor:
    mode = normalize_pooling_mode(pooling_mode)
    if tokens.dim() != 3:
        raise ValueError(f"Patch aggregation expects [B, N, D], got {tuple(tokens.shape)}.")
    if mode == "avg_only":
        return tokens.mean(dim=1)
    if mode == "max_only":
        return tokens.amax(dim=1)
    return torch.cat([tokens.mean(dim=1), tokens.amax(dim=1)], dim=-1)
