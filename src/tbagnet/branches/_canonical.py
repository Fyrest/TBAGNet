from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..defaults import DEFAULT_SPECTRAL_PRB_DEPTH, DEFAULT_TFPATCH_PRB_DEPTH


POOLING_MODES = {"avg_only", "max_only", "dual_pool"}
FREQUENCY_REPRESENTATIONS = {"amplitude_only", "amplitude_phase", "real_imag"}


def _normalize_pooling_mode(pooling_mode: str) -> str:
    mode = str(pooling_mode).lower()
    if mode not in POOLING_MODES:
        raise ValueError("pooling_mode must be avg_only, max_only, or dual_pool.")
    return mode


def _normalize_frequency_representation(frequency_representation: str) -> str:
    aliases = {
        "amplitude_only": "amplitude_only",
        "amplitude": "amplitude_only",
        "magnitude_only": "amplitude_only",
        "amplitude_phase": "amplitude_phase",
        "magnitude_phase": "amplitude_phase",
        "real_imag": "real_imag",
        "real_imaginary": "real_imag",
    }
    mode = aliases.get(str(frequency_representation).lower())
    if mode not in FREQUENCY_REPRESENTATIONS:
        raise ValueError("frequency_representation must be amplitude_only, amplitude_phase, or real_imag.")
    return mode


def _pad_for_patching(x: torch.Tensor, patch_size: int, patch_stride: int) -> torch.Tensor:
    freq_bins = x.size(-1)
    if freq_bins < patch_size:
        pad_right = patch_size - freq_bins
    else:
        remainder = (freq_bins - patch_size) % patch_stride
        pad_right = 0 if remainder == 0 else patch_stride - remainder
    return x if pad_right == 0 else F.pad(x, (0, pad_right))


def _frequency_patch(x: torch.Tensor, patch_size: int, patch_stride: int) -> torch.Tensor:
    x = _pad_for_patching(x, patch_size, patch_stride)
    patches = x.unfold(dimension=-1, size=patch_size, step=patch_stride)
    batch_size, channels, patch_num, _ = patches.shape
    patches = patches.permute(0, 2, 1, 3).contiguous()
    return patches.view(batch_size, patch_num, channels * patch_size)


class TemporalBranch(nn.Module):
    """Temporal waveform branch.

    Input:  x [B, C, T]
    Output: feature [B, D]
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        out_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.GELU(),
            nn.Conv1d(base_channels, base_channels, kernel_size=7, padding=3, groups=base_channels, bias=False),
            nn.Conv1d(base_channels, base_channels * 2, kernel_size=1, bias=False),
            nn.BatchNorm1d(base_channels * 2),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(base_channels * 2, base_channels * 2, kernel_size=7, padding=3, groups=base_channels * 2, bias=False),
            nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=1, bias=False),
            nn.BatchNorm1d(base_channels * 4),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base_channels * 4, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"TemporalBranch expects [B, C, T], got {tuple(x.shape)}.")
        return self.proj(self.encoder(x))


class DepthwisePatchRelationBlock(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")
        self.norm = nn.LayerNorm(d_model)
        self.depthwise = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
        )
        self.pointwise = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        relation = self.norm(x).transpose(1, 2)
        relation = self.depthwise(relation)
        relation = self.pointwise(relation).transpose(1, 2)
        x = residual + relation
        return x + self.ffn(x)


class FrequencyPatchRelationMixer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_layers < 0:
            raise ValueError("num_layers must be non-negative.")
        self.num_layers = int(num_layers)
        self.layers = nn.ModuleList(
            [
                DepthwisePatchRelationBlock(d_model, kernel_size, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model) if self.num_layers > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class SoftFrequencyBandGate(nn.Module):
    def __init__(self, d_model: int, reduction: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        hidden_dim = max(1, d_model // reduction)
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpectralBranch(nn.Module):
    """FFT / spectral patch branch.

    Input:  x [B, C, T]
    Output: feature [B, D]
    """

    def __init__(
        self,
        in_channels: int = 1,
        freq_patch_size: int = 16,
        freq_patch_stride: int = 8,
        frequency_representation: str = "real_imag",
        out_dim: int = 128,
        spectral_relation_depth: int | None = None,
        mixer_layers: int | None = None,
        num_patch_relation_blocks: int | None = None,
        spectral_patch_relation_depth: int | None = DEFAULT_SPECTRAL_PRB_DEPTH,
        mixer_kernel_size: int = 3,
        dropout: float = 0.1,
        use_gate: bool = True,
        gate_reduction: int = 4,
        pooling_mode: str = "max_only",
    ) -> None:
        super().__init__()
        if freq_patch_size <= 0 or freq_patch_stride <= 0:
            raise ValueError("freq_patch_size and freq_patch_stride must be positive.")
        self.freq_patch_size = freq_patch_size
        self.freq_patch_stride = freq_patch_stride
        self.frequency_representation = _normalize_frequency_representation(frequency_representation)
        self.phase_magnitude_epsilon = 1e-8
        self.pooling_mode = _normalize_pooling_mode(pooling_mode)
        self.last_token_shape: tuple[int, ...] | None = None
        self.last_feature_shape: tuple[int, ...] | None = None
        self.last_used_avg_pool = False
        self.last_used_max_pool = False
        relation_depth = spectral_relation_depth
        if relation_depth is None:
            relation_depth = spectral_patch_relation_depth
        if relation_depth is None:
            relation_depth = num_patch_relation_blocks
        if relation_depth is None:
            relation_depth = DEFAULT_SPECTRAL_PRB_DEPTH
        self.spectral_relation_depth = int(relation_depth)
        self.spectral_patch_relation_depth = self.spectral_relation_depth
        self.num_patch_relation_blocks = self.spectral_relation_depth
        if self.spectral_relation_depth < 0:
            raise ValueError("spectral_relation_depth must be non-negative.")
        patch_dim = in_channels * freq_patch_size * 2
        self.patch_embedding = nn.Sequential(
            nn.Linear(patch_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout),
        )
        self.patch_relation_mixer = FrequencyPatchRelationMixer(
            d_model=out_dim,
            num_layers=self.num_patch_relation_blocks,
            kernel_size=mixer_kernel_size,
            dropout=dropout,
        )
        self.frequency_gate = (
            SoftFrequencyBandGate(out_dim, gate_reduction, dropout)
            if use_gate
            else None
        )
        pooled_dim = out_dim * 2 if self.pooling_mode == "dual_pool" else out_dim
        self.out_proj = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def _spectrum_components(self, spectrum: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.frequency_representation == "real_imag":
            return spectrum.real, spectrum.imag
        amplitude = torch.sqrt(spectrum.real.square() + spectrum.imag.square())
        if self.frequency_representation == "amplitude_only":
            return amplitude, torch.zeros_like(amplitude)
        phase = torch.atan2(spectrum.imag, spectrum.real) / torch.pi
        phase = torch.where(amplitude < self.phase_magnitude_epsilon, torch.zeros_like(phase), phase)
        return amplitude, phase

    def _make_patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(x, dim=-1)
        first_component, second_component = self._spectrum_components(spectrum)
        first_patches = _frequency_patch(
            first_component,
            patch_size=self.freq_patch_size,
            patch_stride=self.freq_patch_stride,
        )
        second_patches = _frequency_patch(
            second_component,
            patch_size=self.freq_patch_size,
            patch_stride=self.freq_patch_stride,
        )
        return self.patch_embedding(torch.cat([first_patches, second_patches], dim=-1))

    def _aggregate_patch_tokens(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        self.last_token_shape = tuple(patch_tokens.shape)
        self.last_used_avg_pool = False
        self.last_used_max_pool = False
        if self.pooling_mode == "avg_only":
            self.last_used_avg_pool = True
            return patch_tokens.mean(dim=1)
        if self.pooling_mode == "max_only":
            self.last_used_max_pool = True
            return patch_tokens.amax(dim=1)
        self.last_used_avg_pool = True
        self.last_used_max_pool = True
        mean_pool = patch_tokens.mean(dim=1)
        max_pool = patch_tokens.amax(dim=1)
        return torch.cat([mean_pool, max_pool], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"SpectralBranch expects [B, C, T], got {tuple(x.shape)}.")
        patch_tokens = self._make_patch_tokens(x)
        patch_tokens = self.patch_relation_mixer(patch_tokens)
        if self.frequency_gate is not None:
            patch_tokens = patch_tokens * self.frequency_gate(patch_tokens)
        feature = self.out_proj(self._aggregate_patch_tokens(patch_tokens))
        self.last_feature_shape = tuple(feature.shape)
        return feature


class TimeFrequencyTextureBranch(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        out_dim: int = 128,
        dropout: float = 0.1,
        conv_depth: int = 4,
    ) -> None:
        super().__init__()
        if conv_depth < 1:
            raise ValueError("conv_depth must be at least 1.")
        self.conv_depth = int(conv_depth)
        self.base_channels = int(base_channels)
        blocks: list[nn.Module] = []
        current_channels = int(in_channels)
        for block_idx in range(self.conv_depth):
            if block_idx == 0:
                next_channels = base_channels
            elif block_idx == 1:
                next_channels = base_channels * 2
            else:
                next_channels = base_channels * 4
            blocks.extend(
                [
                    nn.Conv2d(current_channels, next_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(next_channels),
                    nn.GELU(),
                    nn.MaxPool2d(2),
                ]
            )
            current_channels = next_channels
        blocks.extend(
            [
                nn.Conv2d(current_channels, base_channels * 4, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(base_channels * 4),
                nn.GELU(),
                nn.AdaptiveAvgPool2d(1),
            ]
        )
        self.encoder = nn.Sequential(*blocks)
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base_channels * 4, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.encoder(x))


class TimeFrequencyPatchBranch(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        patch_size: int = 16,
        patch_stride: int = 8,
        out_dim: int = 128,
        tfpatch_relation_depth: int | None = None,
        mixer_layers: int | None = None,
        tfpatch_patch_relation_depth: int | None = DEFAULT_TFPATCH_PRB_DEPTH,
        mixer_kernel_size: int = 3,
        dropout: float = 0.1,
        use_gate: bool = True,
        gate_reduction: int = 4,
        pooling_mode: str = "max_only",
    ) -> None:
        super().__init__()
        if patch_size <= 0 or patch_stride <= 0:
            raise ValueError("patch_size and patch_stride must be positive.")
        self.pooling_mode = _normalize_pooling_mode(pooling_mode)
        self.last_token_shape: tuple[int, ...] | None = None
        self.last_feature_shape: tuple[int, ...] | None = None
        self.last_used_avg_pool = False
        self.last_used_max_pool = False
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_stride)
        self.patch_embedding = nn.Sequential(
            nn.Linear(in_channels * patch_size * patch_size, out_dim),
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout),
        )
        relation_depth = tfpatch_relation_depth
        if relation_depth is None:
            relation_depth = tfpatch_patch_relation_depth
        if relation_depth is None:
            relation_depth = DEFAULT_TFPATCH_PRB_DEPTH
        self.tfpatch_relation_depth = int(relation_depth)
        self.tfpatch_patch_relation_depth = self.tfpatch_relation_depth
        if self.tfpatch_relation_depth < 0:
            raise ValueError("tfpatch_relation_depth must be non-negative.")
        relation_layers: list[nn.Module] = [
            DepthwisePatchRelationBlock(out_dim, kernel_size=mixer_kernel_size, dropout=dropout)
            for _ in range(self.tfpatch_relation_depth)
        ]
        if self.tfpatch_relation_depth > 0:
            relation_layers.append(nn.LayerNorm(out_dim))
        self.patch_relation_mixer = nn.Sequential(
            *relation_layers,
        )
        self.patch_gate = (
            SoftFrequencyBandGate(out_dim, reduction=gate_reduction, dropout=dropout)
            if use_gate
            else None
        )
        pooled_dim = out_dim * 2 if self.pooling_mode == "dual_pool" else out_dim
        self.out_proj = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def _aggregate_patch_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        self.last_token_shape = tuple(tokens.shape)
        self.last_used_avg_pool = False
        self.last_used_max_pool = False
        if self.pooling_mode == "avg_only":
            self.last_used_avg_pool = True
            return tokens.mean(dim=1)
        if self.pooling_mode == "max_only":
            self.last_used_max_pool = True
            return tokens.amax(dim=1)
        self.last_used_avg_pool = True
        self.last_used_max_pool = True
        mean_pool = tokens.mean(dim=1)
        max_pool = tokens.amax(dim=1)
        return torch.cat([mean_pool, max_pool], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.unfold(x).transpose(1, 2).contiguous()
        tokens = self.patch_embedding(patches)
        tokens = self.patch_relation_mixer(tokens)
        if self.patch_gate is not None:
            tokens = tokens * self.patch_gate(tokens)
        feature = self.out_proj(self._aggregate_patch_tokens(tokens))
        self.last_feature_shape = tuple(feature.shape)
        return feature


class WaveletBranch(nn.Module):
    """Wavelet / time-frequency branch.

    Input:  raw waveform x [B, C, T]
    Output: feature [B, D]

    The paper model uses a torch-based real Morlet CWT image before the
    local-texture and global-relation encoders.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_dim: int = 128,
        tf_transform: str = "cwt_morl",
        wavelet_mode: str | None = "real_morlet",
        wavelet_subbranch_mode: str = "dual",
        image_size: Tuple[int, int] = (64, 64),
        base_channels: int = 32,
        patch_size: int = 16,
        patch_stride: int = 8,
        tfpatch_relation_depth: int | None = None,
        mixer_layers: int | None = None,
        tfpatch_patch_relation_depth: int | None = DEFAULT_TFPATCH_PRB_DEPTH,
        mixer_kernel_size: int = 3,
        wavelet_conv_depth: int = 4,
        wavelet_scales: int = 64,
        wavelet_kernel_size: int = 129,
        wavelet_center_frequency: float = 6.0,
        kernel_mean_subtraction: bool = True,
        dropout: float = 0.1,
        use_gate: bool = True,
        gate_reduction: int = 4,
        pooling_mode: str = "max_only",
    ) -> None:
        super().__init__()
        mode_aliases = {
            "real_morlet": "real_morlet",
            "complex_morlet": "complex_morlet",
            "cwt_morl": "real_morlet",
            "cwt_cmor": "complex_morlet",
            "mexh": "mexh",
            "gaus1": "gaus1",
            "cgau1": "cgau1",
            "cwt_mexh": "mexh",
            "cwt_gaus1": "gaus1",
            "cwt_cgau1": "cgau1",
        }
        requested_mode = wavelet_mode if wavelet_mode is not None else tf_transform
        if requested_mode not in mode_aliases:
            raise ValueError("wavelet_mode must be real_morlet, complex_morlet, mexh, gaus1, or cgau1.")
        if wavelet_subbranch_mode not in {"texture_only", "patch_only", "dual"}:
            raise ValueError("wavelet_subbranch_mode must be texture_only, patch_only, or dual.")
        self.in_channels = int(in_channels)
        self.wavelet_mode = mode_aliases[requested_mode]
        transform_names = {
            "real_morlet": "cwt_morl",
            "complex_morlet": "cwt_cmor",
            "mexh": "cwt_mexh",
            "gaus1": "cwt_gaus1",
            "cgau1": "cwt_cgau1",
        }
        self.tf_transform = transform_names[self.wavelet_mode]
        self.wavelet_subbranch_mode = wavelet_subbranch_mode
        self.pooling_mode = _normalize_pooling_mode(pooling_mode)
        self.image_size = tuple(int(v) for v in image_size)
        self.wavelet_conv_depth = int(wavelet_conv_depth)
        relation_depth = tfpatch_relation_depth
        if relation_depth is None:
            relation_depth = tfpatch_patch_relation_depth
        if relation_depth is None:
            relation_depth = DEFAULT_TFPATCH_PRB_DEPTH
        self.tfpatch_relation_depth = int(relation_depth)
        self.tfpatch_patch_relation_depth = self.tfpatch_relation_depth
        if self.tfpatch_relation_depth < 0:
            raise ValueError("tfpatch_relation_depth must be non-negative.")
        self.wavelet_scales = int(wavelet_scales)
        self.wavelet_kernel_size = int(wavelet_kernel_size)
        self.wavelet_center_frequency = float(wavelet_center_frequency)
        if self.wavelet_scales <= 0:
            raise ValueError("wavelet_scales must be positive.")
        if self.wavelet_kernel_size <= 0 or self.wavelet_kernel_size % 2 == 0:
            raise ValueError("wavelet_kernel_size must be a positive odd integer.")
        if self.wavelet_center_frequency <= 0:
            raise ValueError("wavelet_center_frequency must be positive.")
        self.wavelet_scale_start = 1
        self.wavelet_scale_end = self.wavelet_scales
        self.kernel_mean_subtraction = bool(kernel_mean_subtraction)
        self.texture_branch = (
            TimeFrequencyTextureBranch(
                in_channels=1,
                base_channels=base_channels,
                out_dim=out_dim,
                dropout=dropout,
                conv_depth=wavelet_conv_depth,
            )
            if self.wavelet_subbranch_mode in {"texture_only", "dual"}
            else None
        )
        self.patch_branch = (
            TimeFrequencyPatchBranch(
                in_channels=1,
                patch_size=patch_size,
                patch_stride=patch_stride,
                out_dim=out_dim,
                tfpatch_relation_depth=self.tfpatch_relation_depth,
                tfpatch_patch_relation_depth=self.tfpatch_relation_depth,
                mixer_kernel_size=mixer_kernel_size,
                dropout=dropout,
                use_gate=use_gate,
                gate_reduction=gate_reduction,
                pooling_mode=self.pooling_mode,
            )
            if self.wavelet_subbranch_mode in {"patch_only", "dual"}
            else None
        )
        self.out_proj = (
            nn.Sequential(
                nn.LayerNorm(out_dim * 2),
                nn.Linear(out_dim * 2, out_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            if self.wavelet_subbranch_mode == "dual"
            else None
        )

    def _check_input(self, x: torch.Tensor) -> None:
        if x.dim() != 3:
            raise ValueError(f"WaveletBranch expects [B, C, T], got {tuple(x.shape)}.")
        if x.size(1) != self.in_channels:
            raise ValueError(f"WaveletBranch expects C={self.in_channels}, got {x.size(1)}.")

    def _raw_to_cwt(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input(x)
        wave = x.mean(dim=1, keepdim=True)
        scales = torch.arange(
            self.wavelet_scale_start,
            self.wavelet_scale_end + 1,
            device=x.device,
            dtype=x.dtype,
        ).view(-1, 1)
        center = (self.wavelet_kernel_size - 1) / 2.0
        t = torch.arange(self.wavelet_kernel_size, device=x.device, dtype=x.dtype).view(1, -1) - center
        u = t / scales
        gaussian = torch.exp(-0.5 * u * u)
        scale_norm = torch.sqrt(scales)
        if self.wavelet_mode in {"real_morlet", "complex_morlet"}:
            real = torch.cos(self.wavelet_center_frequency * u) * gaussian / scale_norm
        elif self.wavelet_mode == "mexh":
            normalization = 2.0 / (3.0**0.5 * torch.pi**0.25)
            real = normalization * (1.0 - u.square()) * gaussian / scale_norm
        elif self.wavelet_mode == "gaus1":
            gaus1_base = (2.0 / torch.pi) ** 0.25 * torch.exp(-u.square())
            real = -2.0 * u * gaus1_base / scale_norm
        else:
            cgau_base = torch.exp(-u.square())
            cgau_carrier_real = torch.cos(u)
            cgau_carrier_imag = -torch.sin(u)
            cgau_normalization = (
                torch.exp(u.new_tensor(-0.5)) * (2.0**0.5) * (torch.pi**0.5)
            ) ** 0.5
            cgau_base = cgau_base / cgau_normalization
            real = (2.0**0.5) * cgau_base * (
                -2.0 * u * cgau_carrier_real + cgau_carrier_imag
            ) / scale_norm
            imag = (2.0**0.5) * cgau_base * (
                -2.0 * u * cgau_carrier_imag - cgau_carrier_real
            ) / scale_norm
        if self.kernel_mean_subtraction:
            real = real - real.mean(dim=1, keepdim=True)
        real_weight = real.unsqueeze(1)
        real_feat = F.conv1d(wave, real_weight, padding=self.wavelet_kernel_size // 2)
        if self.wavelet_mode in {"real_morlet", "mexh", "gaus1"}:
            image = torch.log1p(real_feat.abs()).unsqueeze(1)
        else:
            if self.wavelet_mode == "complex_morlet":
                imag = torch.sin(self.wavelet_center_frequency * u) * gaussian / scale_norm
            if self.kernel_mean_subtraction:
                imag = imag - imag.mean(dim=1, keepdim=True)
            imag_weight = imag.unsqueeze(1)
            imag_feat = F.conv1d(wave, imag_weight, padding=self.wavelet_kernel_size // 2)
            image = torch.log1p(torch.sqrt(real_feat.square() + imag_feat.square() + 1e-8)).unsqueeze(1)
        return F.interpolate(image, size=self.image_size, mode="bilinear", align_corners=False)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return self._raw_to_cwt(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tf_image = self.transform(x)
        if self.wavelet_subbranch_mode == "texture_only":
            if self.texture_branch is None:
                raise RuntimeError("texture_only mode requires texture_branch.")
            return self.texture_branch(tf_image)
        if self.wavelet_subbranch_mode == "patch_only":
            if self.patch_branch is None:
                raise RuntimeError("patch_only mode requires patch_branch.")
            return self.patch_branch(tf_image)
        if self.texture_branch is None or self.patch_branch is None or self.out_proj is None:
            raise RuntimeError("dual mode requires texture_branch, patch_branch, and out_proj.")
        texture_feat = self.texture_branch(tf_image)
        patch_feat = self.patch_branch(tf_image)
        return self.out_proj(torch.cat([texture_feat, patch_feat], dim=-1))


FFTBranch = SpectralBranch
CWTBranch = WaveletBranch

