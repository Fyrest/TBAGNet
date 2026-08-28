from __future__ import annotations

import warnings
from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn

from .branches import SpectralBranch, TemporalBranch, WaveletBranch
from .defaults import DEFAULT_SPECTRAL_PRB_DEPTH, DEFAULT_TFPATCH_PRB_DEPTH
from .modules.aga import (
    AddFusion,
    AdaptiveGatedAggregation,
    BRANCH_ORDER,
    ConcatFusion,
    IdentityFusion,
    SoftmaxFusion,
)


BRANCH_SET_TO_VIEWS = {
    "T": ("time",),
    "F": ("fft",),
    "W": ("cwt",),
    "T+F": ("time", "fft"),
    "T+W": ("time", "cwt"),
    "F+W": ("fft", "cwt"),
    "T+F+W": ("time", "fft", "cwt"),
}


def normalize_branch_set(branch_set: str | Iterable[str]) -> Tuple[str, ...]:
    if isinstance(branch_set, str):
        key = branch_set.strip().upper().replace(" ", "")
        aliases = {
            "TIME": "T",
            "FFT": "F",
            "CWT": "W",
            "TEMPORAL": "T",
            "SPECTRAL": "F",
            "WAVELET": "W",
            "TIME+FFT": "T+F",
            "TIME+CWT": "T+W",
            "FFT+CWT": "F+W",
            "TEMPORAL+SPECTRAL": "T+F",
            "TEMPORAL+WAVELET": "T+W",
            "SPECTRAL+WAVELET": "F+W",
            "TIME+FFT+CWT": "T+F+W",
            "TEMPORAL+SPECTRAL+WAVELET": "T+F+W",
        }
        key = aliases.get(key, key)
        if key not in BRANCH_SET_TO_VIEWS:
            raise ValueError(f"Unsupported branch_set={branch_set}.")
        return BRANCH_SET_TO_VIEWS[key]
    view_aliases = {
        "temporal": "time",
        "spectral": "fft",
        "wavelet": "cwt",
        "time": "time",
        "fft": "fft",
        "cwt": "cwt",
    }
    views = tuple(view_aliases.get(str(view).strip().lower(), str(view).strip().lower()) for view in branch_set)
    invalid = [view for view in views if view not in BRANCH_ORDER]
    if invalid:
        raise ValueError(f"Unsupported branch names: {invalid}.")
    return views


class TBAGNet(nn.Module):
    """TBAGNet model with configurable temporal/spectral/wavelet branches.

    Single branch: identity/fixed_reference.
    Two or three branches: Adaptive Gated Aggregation (AGA).
    """

    def __init__(
        self,
        num_classes: int,
        input_channels: int = 1,
        branch_dim: int = 128,
        branch_set: str = "T+F+W",
        fusion_type: str = "aga",
        dropout: float = 0.1,
        time_base_channels: int = 32,
        tf_transform: str = "cwt_morl",
        wavelet_mode: str | None = "real_morlet",
        wavelet_subbranch_mode: str = "dual",
        tf_image_size: Tuple[int, int] = (64, 64),
        freq_patch_size: int = 16,
        freq_patch_stride: int = 8,
        patch_size: int = 16,
        patch_stride: int = 8,
        spectral_relation_depth: int | None = None,
        tfpatch_relation_depth: int | None = None,
        mixer_layers: int | None = None,
        num_patch_relation_blocks: int | None = None,
        spectral_patch_relation_depth: int | None = DEFAULT_SPECTRAL_PRB_DEPTH,
        tfpatch_patch_relation_depth: int | None = DEFAULT_TFPATCH_PRB_DEPTH,
        wavelet_conv_depth: int = 4,
        wavelet_scales: int = 64,
        wavelet_kernel_size: int = 129,
        wavelet_center_frequency: float = 6.0,
        kernel_mean_subtraction: bool = True,
        pooling_mode: str = "max_only",
        mixer_kernel_size: int = 3,
        use_gate: bool = True,
        gate_reduction: int = 4,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.input_channels = int(input_channels)
        self.branch_dim = int(branch_dim)
        self.enabled_views = normalize_branch_set(branch_set)
        self.branch_set = branch_set
        fusion_type = str(fusion_type).lower()
        if fusion_type == "channel_gate":
            warnings.warn(
                "fusion_type='channel_gate' is a compatibility alias; use 'aga'.",
                DeprecationWarning,
                stacklevel=2,
            )
            fusion_type = "aga"
        self.requested_fusion_type = fusion_type
        resolved_spectral_depth = spectral_relation_depth
        if resolved_spectral_depth is None:
            resolved_spectral_depth = spectral_patch_relation_depth
        if resolved_spectral_depth is None:
            resolved_spectral_depth = num_patch_relation_blocks
        if resolved_spectral_depth is None:
            resolved_spectral_depth = DEFAULT_SPECTRAL_PRB_DEPTH
        resolved_tfpatch_depth = tfpatch_relation_depth
        if resolved_tfpatch_depth is None:
            resolved_tfpatch_depth = tfpatch_patch_relation_depth
        if resolved_tfpatch_depth is None:
            resolved_tfpatch_depth = DEFAULT_TFPATCH_PRB_DEPTH
        self.spectral_relation_depth = int(resolved_spectral_depth)
        self.tfpatch_relation_depth = int(resolved_tfpatch_depth)
        self.spectral_patch_relation_depth = self.spectral_relation_depth
        self.tfpatch_patch_relation_depth = self.tfpatch_relation_depth
        self.num_patch_relation_blocks = self.spectral_relation_depth
        self.wavelet_conv_depth = int(wavelet_conv_depth)
        self.kernel_mean_subtraction = bool(kernel_mean_subtraction)
        self.pooling_mode = str(pooling_mode).lower()

        if len(self.enabled_views) == 1 and fusion_type == "aga":
            warnings.warn(
                "single branch has no multi-branch fusion; using identity/fixed_reference.",
                stacklevel=2,
            )
            fusion_type = "fixed_reference"
        if len(self.enabled_views) > 1 and fusion_type in {"identity", "fixed_reference"}:
            raise ValueError("identity/fixed_reference is only valid for a single enabled branch.")
        if fusion_type not in {"identity", "fixed_reference", "aga", "concat", "add", "softmax"}:
            raise ValueError("fusion_type must be identity, fixed_reference, aga, concat, add, or softmax.")
        if fusion_type in {"concat", "add", "softmax"} and self.enabled_views != BRANCH_ORDER:
            raise ValueError(f"{fusion_type} fusion requires the full T+F+W branch set.")
        self.fusion_type = fusion_type

        self.time_branch = TemporalBranch(
            in_channels=input_channels,
            base_channels=time_base_channels,
            out_dim=branch_dim,
            dropout=dropout,
        )
        self.fft_branch = SpectralBranch(
            in_channels=input_channels,
            freq_patch_size=freq_patch_size,
            freq_patch_stride=freq_patch_stride,
            out_dim=branch_dim,
            spectral_relation_depth=self.spectral_relation_depth,
            num_patch_relation_blocks=self.num_patch_relation_blocks,
            spectral_patch_relation_depth=self.spectral_patch_relation_depth,
            mixer_kernel_size=mixer_kernel_size,
            dropout=dropout,
            use_gate=use_gate,
            gate_reduction=gate_reduction,
            pooling_mode=self.pooling_mode,
        )
        self.cwt_branch = WaveletBranch(
            in_channels=input_channels,
            out_dim=branch_dim,
            tf_transform=tf_transform,
            wavelet_mode=wavelet_mode,
            wavelet_subbranch_mode=wavelet_subbranch_mode,
            image_size=tf_image_size,
            base_channels=time_base_channels,
            patch_size=patch_size,
            patch_stride=patch_stride,
            tfpatch_relation_depth=self.tfpatch_relation_depth,
            tfpatch_patch_relation_depth=self.tfpatch_patch_relation_depth,
            mixer_kernel_size=mixer_kernel_size,
            wavelet_conv_depth=self.wavelet_conv_depth,
            wavelet_scales=wavelet_scales,
            wavelet_kernel_size=wavelet_kernel_size,
            wavelet_center_frequency=wavelet_center_frequency,
            kernel_mean_subtraction=self.kernel_mean_subtraction,
            dropout=dropout,
            use_gate=use_gate,
            gate_reduction=gate_reduction,
            pooling_mode=self.pooling_mode,
        )
        self.identity_fusion = IdentityFusion()
        if self.fusion_type == "aga":
            self.aga = AdaptiveGatedAggregation(hidden_dim=branch_dim, dropout=dropout)
        elif self.fusion_type == "concat":
            self.concat_fusion = ConcatFusion(hidden_dim=branch_dim)
        elif self.fusion_type == "add":
            self.add_fusion = AddFusion(hidden_dim=branch_dim)
        elif self.fusion_type == "softmax":
            self.softmax_fusion = SoftmaxFusion(hidden_dim=branch_dim, dropout=dropout)
        classifier_input_dim = branch_dim if self.fusion_type in {"identity", "fixed_reference", "aga", "add"} else branch_dim * len(self.enabled_views)
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_input_dim),
            nn.Dropout(dropout),
            nn.Linear(classifier_input_dim, branch_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(branch_dim, num_classes),
        )

    def count_parameters(self) -> int:
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def extract_branch_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.dim() != 3:
            raise ValueError(f"TBAGNet expects [B, C, T], got {tuple(x.shape)}.")
        if x.size(1) != self.input_channels:
            raise ValueError(f"Expected input_channels={self.input_channels}, got {x.size(1)}.")
        features: Dict[str, torch.Tensor] = {}
        if "time" in self.enabled_views:
            features["time"] = self.time_branch(x)
        if "fft" in self.enabled_views:
            features["fft"] = self.fft_branch(x)
        if "cwt" in self.enabled_views:
            features["cwt"] = self.cwt_branch(x)
        return features

    def fuse_features(self, features: Dict[str, torch.Tensor]):
        if len(self.enabled_views) == 1:
            fused, aux = self.identity_fusion(features, self.enabled_views, return_weights=True)
            aux["fusion_type"] = self.fusion_type
            return fused, aux
        if self.fusion_type == "aga":
            fused, aux = self.aga(features, self.enabled_views, return_weights=True)
        elif self.fusion_type == "concat":
            fused, aux = self.concat_fusion(features, self.enabled_views, return_weights=True)
        elif self.fusion_type == "add":
            fused, aux = self.add_fusion(features, self.enabled_views, return_weights=True)
        elif self.fusion_type == "softmax":
            fused, aux = self.softmax_fusion(features, self.enabled_views, return_weights=True)
        else:
            raise ValueError(f"Unsupported fusion_type={self.fusion_type}.")
        aux["fusion_type"] = self.fusion_type
        return fused, aux

    def forward(self, x: torch.Tensor, return_features: bool = False):
        features = self.extract_branch_features(x)
        fused, aux = self.fuse_features(features)
        logits = self.classifier(fused)
        if return_features:
            return {
                "logits": logits,
                "features": features,
                "fused": fused,
                "enabled_views": self.enabled_views,
                "fusion_type": aux.get("fusion_type", self.fusion_type),
                "branch_weights": aux.get("branch_weights"),
                "branch_logits": aux.get("branch_logits"),
                "channel_weights": aux.get("channel_weights"),
                "padded_channel_weights": aux.get("padded_channel_weights"),
                "padded_branch_weights": aux.get("padded_branch_weights"),
                "fusion_aux": aux,
            }
        return logits

def build_model(config: dict) -> TBAGNet:
    model_cfg = config.get("model", {})
    dataset_cfg = config.get("dataset", {})
    tf_image_size = model_cfg.get("tf_image_size", [64, 64])
    enabled_branches = model_cfg.get("enabled_branches", config.get("enabled_branches"))
    branch_set = enabled_branches if enabled_branches is not None else model_cfg.get("branches", model_cfg.get("branch_set", "T+F+W"))
    return TBAGNet(
        num_classes=int(model_cfg.get("num_classes", dataset_cfg.get("num_classes", 6))),
        input_channels=int(model_cfg.get("input_channels", dataset_cfg.get("input_channels", 1))),
        branch_dim=int(model_cfg.get("feature_dim", model_cfg.get("branch_dim", 128))),
        branch_set=branch_set,
        fusion_type=str(model_cfg.get("fusion_type", "aga")),
        dropout=float(model_cfg.get("dropout", 0.1)),
        tf_transform=str(model_cfg.get("tf_transform", "cwt_morl")),
        wavelet_mode=model_cfg.get("wavelet_mode", "real_morlet"),
        wavelet_subbranch_mode=str(model_cfg.get("wavelet_subbranch_mode", "dual")),
        tf_image_size=(int(tf_image_size[0]), int(tf_image_size[1])),
        freq_patch_size=int(model_cfg.get("freq_patch_size", 16)),
        freq_patch_stride=int(model_cfg.get("freq_patch_stride", 8)),
        patch_size=int(model_cfg.get("tf_patch_size", model_cfg.get("patch_size", 16))),
        patch_stride=int(model_cfg.get("tf_patch_stride", model_cfg.get("patch_stride", 8))),
        spectral_relation_depth=int(model_cfg.get("spectral_relation_depth", 4)),
        tfpatch_relation_depth=int(model_cfg.get("tfpatch_relation_depth", 1)),
        spectral_patch_relation_depth=int(model_cfg.get("spectral_patch_relation_depth", model_cfg.get("spectral_relation_depth", 4))),
        tfpatch_patch_relation_depth=int(model_cfg.get("tfpatch_patch_relation_depth", model_cfg.get("tfpatch_relation_depth", 1))),
        wavelet_conv_depth=int(model_cfg.get("wavelet_conv_depth", 4)),
        wavelet_scales=int(model_cfg.get("wavelet_scales", 64)),
        wavelet_kernel_size=int(model_cfg.get("wavelet_kernel_size", 129)),
        wavelet_center_frequency=float(model_cfg.get("wavelet_center_frequency", 6.0)),
        kernel_mean_subtraction=bool(model_cfg.get("kernel_mean_subtraction", True)),
        pooling_mode=str(model_cfg.get("pooling_mode", "max_only")),
        mixer_kernel_size=int(model_cfg.get("mixer_kernel_size", 3)),
        use_gate=bool(model_cfg.get("use_gate", True)),
        gate_reduction=int(model_cfg.get("gate_reduction", 4)),
    )
