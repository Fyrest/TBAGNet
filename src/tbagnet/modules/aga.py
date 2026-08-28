from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn


BRANCH_ORDER = ("time", "fft", "cwt")


def _branch_normalize(logits: torch.Tensor, dim: int) -> torch.Tensor:
    shifted = logits - logits.amax(dim=dim, keepdim=True)
    exp = torch.exp(shifted)
    return exp / exp.sum(dim=dim, keepdim=True).clamp_min(torch.finfo(exp.dtype).eps)


class IdentityFusion(nn.Module):
    """Pass-through fusion used by single-branch experiments."""

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        enabled_views: Iterable[str],
        return_weights: bool = False,
    ):
        views = tuple(enabled_views)
        if len(views) != 1:
            raise ValueError("IdentityFusion expects exactly one enabled branch.")
        fused = features[views[0]]
        aux = {"enabled_views": views}
        if return_weights:
            return fused, aux
        return fused


FixedReferenceFusion = IdentityFusion


class AddFusion(nn.Module):
    """Layer-normalize T/F/W features independently, then add them."""

    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.norms = nn.ModuleDict({name: nn.LayerNorm(hidden_dim) for name in BRANCH_ORDER})

    def _validate(
        self,
        features: Dict[str, torch.Tensor],
        enabled_views: Iterable[str],
    ) -> Tuple[str, ...]:
        views = tuple(enabled_views)
        if views != BRANCH_ORDER:
            raise ValueError(f"AddFusion expects enabled views {BRANCH_ORDER}, got {views}.")
        missing = [name for name in views if name not in features or features[name] is None]
        if missing:
            raise ValueError(f"Missing feature tensors for enabled views: {missing}.")
        ref = features[views[0]]
        if ref.dim() != 2 or ref.size(-1) != self.hidden_dim:
            raise ValueError(f"Expected [B, {self.hidden_dim}], got {tuple(ref.shape)}.")
        for name in views:
            if features[name].shape != ref.shape:
                raise ValueError(f"Feature shape mismatch for {name}: {tuple(features[name].shape)} vs {tuple(ref.shape)}.")
        return views

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        enabled_views: Iterable[str],
        return_weights: bool = False,
    ):
        views = self._validate(features, enabled_views)
        fused = sum((self.norms[name](features[name]) for name in views))
        aux = {"enabled_views": views, "fusion_type": "add"}
        if return_weights:
            return fused, aux
        return fused


class AdaptiveGatedAggregation(nn.Module):
    """Adaptive Gated Aggregation (AGA) from the TBAGNet paper.

    Input features are [B, D] tensors from enabled branches. For N enabled
    branches, the active weights have shape [B, N, D]. Normalization is applied
    along the branch dimension for every channel, so disabled branches never
    receive weights and cannot affect the fused feature.
    """

    def __init__(self, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.norms = nn.ModuleDict({name: nn.LayerNorm(hidden_dim) for name in BRANCH_ORDER})
        self.gate_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim * len(BRANCH_ORDER)),
            nn.Linear(hidden_dim * len(BRANCH_ORDER), hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim * len(BRANCH_ORDER)),
        )

    def _validate(
        self,
        features: Dict[str, torch.Tensor],
        enabled_views: Iterable[str],
    ) -> Tuple[str, ...]:
        views = tuple(enabled_views)
        if len(views) < 2:
            raise ValueError("AdaptiveGatedAggregation requires two or more enabled branches.")
        invalid = [name for name in views if name not in BRANCH_ORDER]
        if invalid:
            raise ValueError(f"Unsupported enabled views: {invalid}.")
        missing = [name for name in views if name not in features or features[name] is None]
        if missing:
            raise ValueError(f"Missing feature tensors for enabled views: {missing}.")
        ref = features[views[0]]
        if ref.dim() != 2 or ref.size(-1) != self.hidden_dim:
            raise ValueError(f"Expected [B, {self.hidden_dim}], got {tuple(ref.shape)}.")
        for name in views:
            if features[name].shape != ref.shape:
                raise ValueError(f"Feature shape mismatch for {name}: {tuple(features[name].shape)} vs {tuple(ref.shape)}.")
        return views

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        enabled_views: Iterable[str],
        return_weights: bool = False,
    ):
        views = self._validate(features, enabled_views)
        ref = features[views[0]]
        normalized = {name: self.norms[name](features[name]) for name in views}
        zero = ref.new_zeros(ref.shape)
        gate_input = torch.cat([normalized.get(name, zero) for name in BRANCH_ORDER], dim=-1)
        logits = self.gate_mlp(gate_input).view(-1, len(BRANCH_ORDER), self.hidden_dim)

        active_indices = [BRANCH_ORDER.index(name) for name in views]
        active_logits = logits[:, active_indices, :]
        active_channel_weights = _branch_normalize(active_logits, dim=1)
        active_tokens = torch.stack([normalized[name] for name in views], dim=1)
        fused = (active_channel_weights * active_tokens).sum(dim=1)

        padded_channel_weights = logits.new_zeros(logits.shape)
        for active_pos, branch_idx in enumerate(active_indices):
            padded_channel_weights[:, branch_idx, :] = active_channel_weights[:, active_pos, :]
        branch_weights = active_channel_weights.mean(dim=-1)
        aux = {
            "enabled_views": views,
            "channel_weights": active_channel_weights,
            "padded_channel_weights": padded_channel_weights,
            "branch_weights": branch_weights,
            "padded_branch_weights": padded_channel_weights.mean(dim=-1),
            "fusion_logits": logits,
        }
        if return_weights:
            return fused, aux
        return fused


ChannelGateFusion = AdaptiveGatedAggregation


class ConcatFusion(nn.Module):
    """Direct feature concatenation for T/F/W branches."""

    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)

    def _validate(
        self,
        features: Dict[str, torch.Tensor],
        enabled_views: Iterable[str],
    ) -> Tuple[str, ...]:
        views = tuple(enabled_views)
        if views != BRANCH_ORDER:
            raise ValueError(f"ConcatFusion expects enabled views {BRANCH_ORDER}, got {views}.")
        missing = [name for name in views if name not in features or features[name] is None]
        if missing:
            raise ValueError(f"Missing feature tensors for enabled views: {missing}.")
        ref = features[views[0]]
        if ref.dim() != 2 or ref.size(-1) != self.hidden_dim:
            raise ValueError(f"Expected [B, {self.hidden_dim}], got {tuple(ref.shape)}.")
        for name in views:
            if features[name].shape != ref.shape:
                raise ValueError(f"Feature shape mismatch for {name}: {tuple(features[name].shape)} vs {tuple(ref.shape)}.")
        return views

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        enabled_views: Iterable[str],
        return_weights: bool = False,
    ):
        views = self._validate(features, enabled_views)
        fused = torch.cat([features[name] for name in views], dim=-1)
        aux = {"enabled_views": views, "fusion_type": "concat"}
        if return_weights:
            return fused, aux
        return fused


class SoftmaxFusion(nn.Module):
    """Branch-level softmax adaptive weighting for T/F/W branches."""

    def __init__(self, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.weight_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim * len(BRANCH_ORDER)),
            nn.Linear(hidden_dim * len(BRANCH_ORDER), hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, len(BRANCH_ORDER)),
        )

    def _validate(
        self,
        features: Dict[str, torch.Tensor],
        enabled_views: Iterable[str],
    ) -> Tuple[str, ...]:
        views = tuple(enabled_views)
        if views != BRANCH_ORDER:
            raise ValueError(f"SoftmaxFusion expects enabled views {BRANCH_ORDER}, got {views}.")
        missing = [name for name in views if name not in features or features[name] is None]
        if missing:
            raise ValueError(f"Missing feature tensors for enabled views: {missing}.")
        ref = features[views[0]]
        if ref.dim() != 2 or ref.size(-1) != self.hidden_dim:
            raise ValueError(f"Expected [B, {self.hidden_dim}], got {tuple(ref.shape)}.")
        for name in views:
            if features[name].shape != ref.shape:
                raise ValueError(f"Feature shape mismatch for {name}: {tuple(features[name].shape)} vs {tuple(ref.shape)}.")
        return views

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        enabled_views: Iterable[str],
        return_weights: bool = False,
    ):
        views = self._validate(features, enabled_views)
        branch_features = [features[name] for name in views]
        fusion_input = torch.cat(branch_features, dim=-1)
        branch_logits = self.weight_mlp(fusion_input)
        branch_weights = torch.softmax(branch_logits, dim=1)
        weighted = [
            feature * branch_weights[:, idx:idx + 1]
            for idx, feature in enumerate(branch_features)
        ]
        fused = torch.cat(weighted, dim=-1)
        aux = {
            "enabled_views": views,
            "fusion_type": "softmax",
            "branch_logits": branch_logits,
            "branch_weights": branch_weights,
            "padded_branch_weights": branch_weights,
        }
        if return_weights:
            return fused, aux
        return fused
