from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import os
import platform
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

from data.dataset_adapters import build_dataloaders, read_label_names
from data.split_utils import sha256_file
from tbagnet.model import build_model
from tbagnet.utils.checkpoint import checkpoint_state_dict, load_checkpoint_payload, save_checkpoint
from tbagnet.utils.logging import setup_logging
from tbagnet.utils.metrics import (
    compute_classification_metrics,
    plot_confusion_matrix,
    prediction_hash,
    save_confusion_matrix_csv,
    save_per_class_metrics,
    save_prediction_tables,
    softmax_probabilities,
)
from tbagnet.utils.seed import seed_everything


ROOT = Path(__file__).resolve().parents[2]
SOURCE_HASH_PATHS = [
    "src/tbagnet/model.py",
    "src/tbagnet/defaults.py",
    "src/tbagnet/branches/_canonical.py",
    "src/tbagnet/modules/aga.py",
    "src/tbagnet/modules/patch_aggregation.py",
    "src/tbagnet/modules/patch_interaction.py",
    "src/tbagnet/utils/checkpoint.py",
    "src/tbagnet/utils/metrics.py",
    "src/tbagnet/utils/seed.py",
    "src/tbagnet/training_runner.py",
    "data/dataset_adapters.py",
    "data/preprocessing.py",
    "data/split_utils.py",
    "train.py",
    "test.py",
]


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def write_yaml(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(to_jsonable(payload), allow_unicode=True, sort_keys=False), encoding="utf-8")


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def resolve_under_root(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (ROOT / path).resolve()


def public_path(path: str | Path) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return "<external>"


def redacted_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(config)
    record_local_paths = bool(cfg.get("experiment", {}).get("record_local_paths", False))
    if "dataset" in cfg and "data_root" in cfg["dataset"] and not record_local_paths:
        cfg["dataset"]["data_root"] = "<DATASET_ROOT>"
    for section in ("output",):
        if section in cfg:
            for key, value in list(cfg[section].items()):
                cfg[section][key] = public_path(value) if str(value) else value
    return cfg


def redact_local_paths(text: str) -> str:
    text = str(text)
    text = text.replace("\\", "/")
    text = re.sub(r"[A-Za-z]:/[^ \n\r\t]+", "<LOCAL_PATH>", text)
    return text


def source_hashes(extra_paths: list[str] | None = None) -> dict[str, dict[str, Any]]:
    paths = list(dict.fromkeys(SOURCE_HASH_PATHS + (extra_paths or [])))
    out: dict[str, dict[str, Any]] = {}
    for item in paths:
        path = ROOT / item
        if path.exists() and path.is_file():
            out[item] = {
                "sha256": sha256_file(path),
                "size": int(path.stat().st_size),
                "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
            }
    return out


def environment_report() -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "python_executable": "<python>",
        "platform": platform.platform(),
        "cwd": "<release_root>",
        "PYTHONPATH": "<redacted>" if os.environ.get("PYTHONPATH") else None,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "torch_version": torch.__version__,
        "cuda_available": bool(cuda_available),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
        "cuda_devices": [
            {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "capability": torch.cuda.get_device_capability(i),
            }
            for i in range(torch.cuda.device_count())
        ]
        if cuda_available
        else [],
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
    }


def resolve_config(config: dict[str, Any], data_root: str | Path | None = None, output_dir: str | Path | None = None) -> dict[str, Any]:
    cfg = copy.deepcopy(config)
    cfg.setdefault("experiment", {})
    cfg.setdefault("dataset", {})
    cfg.setdefault("model", {})
    cfg.setdefault("training", {})
    cfg.setdefault("output", {})
    dataset = cfg["dataset"]
    model = cfg["model"]
    training = cfg["training"]
    output = cfg["output"]

    dataset.setdefault("name", dataset.get("dataset_name", "Railway-DAS"))
    dataset.setdefault("dataset_name", dataset["name"])
    dataset.setdefault("input_length", 1024)
    dataset.setdefault("input_channels", 1)
    dataset.setdefault("num_classes", 6)
    dataset.setdefault("normalize", True)
    dataset.setdefault("data_key", "signal")
    dataset.setdefault("mat_cache_size", 8)
    dataset.setdefault("split_seed", 42)
    if str(dataset["name"]).lower().replace("_", "-") in {"railway-das", "railway das"}:
        dataset.setdefault("split_dir", "data/splits/railway_das")
    if data_root is not None:
        dataset["data_root"] = str(Path(data_root).resolve())

    model.setdefault("name", "TBAGNet")
    model.setdefault("branches", model.get("branch_set", "T+F+W"))
    model.setdefault("branch_set", model["branches"])
    model.setdefault("feature_dim", model.get("branch_dim", 128))
    model.setdefault("branch_dim", model["feature_dim"])
    model.setdefault("fusion_type", "aga")
    model.setdefault("input_channels", dataset["input_channels"])
    model.setdefault("num_classes", dataset["num_classes"])
    model.setdefault("dropout", 0.1)
    model.setdefault("tf_transform", "cwt_morl")
    model.setdefault("wavelet_mode", "real_morlet")
    model.setdefault("wavelet_subbranch_mode", "dual")
    model.setdefault("tf_image_size", [64, 64])
    model.setdefault("wavelet_scales", 64)
    model.setdefault("wavelet_kernel_size", 129)
    model.setdefault("wavelet_center_frequency", 6.0)
    model.setdefault("wavelet_conv_depth", 4)
    model.setdefault("kernel_mean_subtraction", True)
    model.setdefault("spectral_relation_depth", 4)
    model.setdefault("tfpatch_relation_depth", 1)
    model.setdefault("spectral_patch_relation_depth", model["spectral_relation_depth"])
    model.setdefault("tfpatch_patch_relation_depth", model["tfpatch_relation_depth"])
    model.setdefault("pooling_mode", "max_only")
    model.setdefault("freq_patch_size", 16)
    model.setdefault("freq_patch_stride", 8)
    model.setdefault("patch_size", 16)
    model.setdefault("patch_stride", 8)
    model.setdefault("tf_patch_size", model["patch_size"])
    model.setdefault("tf_patch_stride", model["patch_stride"])
    model.setdefault("mixer_kernel_size", 3)
    model.setdefault("use_gate", True)
    model.setdefault("gate_reduction", 4)
    if "mixer_layers" in model:
        raise ValueError("mixer_layers is not allowed in the release config; use spectral_relation_depth and tfpatch_relation_depth.")

    training.setdefault("seed", training.get("run_seed", 42))
    training.setdefault("run_seed", training["seed"])
    training.setdefault("split_seed", dataset["split_seed"])
    training.setdefault("batch_size", 64)
    training.setdefault("learning_rate", 0.001)
    training.setdefault("weight_decay", 0.0001)
    training.setdefault("max_epochs", 80)
    training.setdefault("patience", 12)
    training.setdefault("optimizer", "AdamW")
    training.setdefault("loss", "CrossEntropyLoss")
    training.setdefault("scheduler", None)
    training.setdefault("class_weights", None)
    training.setdefault("num_workers", 0)
    training.setdefault("pin_memory", False)
    training.setdefault("persistent_workers", False)
    training.setdefault("drop_last", False)
    training.setdefault("amp", False)
    training.setdefault("deterministic", True)
    training.setdefault("cudnn_benchmark", False)
    training.setdefault("cudnn_deterministic", True)
    training.setdefault("checkpoint_selection", "val_macro_f1")
    training.setdefault("resume", False)
    training.setdefault("pretrained_checkpoint", None)
    if bool(training["resume"]) or training["pretrained_checkpoint"]:
        raise ValueError("Release reproduction starts from scratch; resume/pretrained_checkpoint are disabled.")
    if bool(training["amp"]):
        raise ValueError("Canonical Railway-DAS reference uses AMP=False.")

    cfg["experiment"].setdefault("name", "railway_tbagnet_real_morlet_maxpool_seed42")
    cfg["experiment"].setdefault("protocol", "legacy_patch00_protocol")
    cfg.setdefault("device", "cuda:0" if torch.cuda.is_available() else "cpu")
    if output_dir is not None:
        output["result_dir"] = str(output_dir)
    output.setdefault("result_dir", "results/reproduced/railway_seed42")
    result_dir = resolve_under_root(output["result_dir"])
    output["result_dir"] = str(result_dir)
    output["checkpoint_dir"] = str(resolve_under_root(output.get("checkpoint_dir", result_dir / "checkpoint")))
    output["log_dir"] = str(resolve_under_root(output.get("log_dir", result_dir)))
    cfg["rng_protocol"] = (
        "seed_everything(seed) -> DataLoaders(generator=seed, worker_init_fn) -> model -> optimizer -> "
        "estimate_macs(torch.randn/model forward) -> training"
    )
    return cfg


def build_criterion(config: dict[str, Any], device: torch.device) -> nn.Module:
    training = config["training"]
    if str(training.get("loss", "CrossEntropyLoss")) != "CrossEntropyLoss":
        raise ValueError("Only CrossEntropyLoss is supported by the release runner.")
    weights = training.get("class_weights")
    if weights is not None:
        return nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    return nn.CrossEntropyLoss()


def build_optimizer(config: dict[str, Any], model: nn.Module) -> torch.optim.Optimizer:
    training = config["training"]
    if str(training.get("optimizer", "AdamW")) != "AdamW":
        raise ValueError("Only AdamW is supported by the release runner.")
    return torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(training.get("learning_rate", 0.001)),
        weight_decay=float(training.get("weight_decay", 0.0001)),
    )


def state_dict_hash(model: nn.Module) -> str:
    h = hashlib.sha256()
    for key, tensor in model.state_dict().items():
        arr = tensor.detach().cpu().contiguous().numpy()
        h.update(key.encode("utf-8"))
        h.update(str(arr.shape).encode("utf-8"))
        h.update(str(arr.dtype).encode("utf-8"))
        h.update(arr.tobytes())
    return h.hexdigest()


def count_params(model: nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters() if param.requires_grad))


def estimate_macs(
    model: nn.Module,
    device: torch.device,
    input_length: int = 1024,
    input_channels: int = 1,
) -> dict[str, Any]:
    macs: dict[str, int] = {"Conv1d": 0, "Conv2d": 0, "Linear": 0, "CWT_functional": 0}
    hooks = []

    def hook(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        if isinstance(output, (tuple, list)):
            return
        if isinstance(module, nn.Conv1d):
            batch, out_channels, out_len = output.shape
            macs["Conv1d"] += int(batch * out_channels * out_len * (module.in_channels // module.groups) * module.kernel_size[0])
        elif isinstance(module, nn.Conv2d):
            batch, out_channels, out_h, out_w = output.shape
            k_h, k_w = module.kernel_size
            macs["Conv2d"] += int(batch * out_channels * out_h * out_w * (module.in_channels // module.groups) * k_h * k_w)
        elif isinstance(module, nn.Linear):
            items = output.shape[0] if output.dim() == 2 else int(np.prod(output.shape[:-1]))
            macs["Linear"] += int(items * module.in_features * module.out_features)

    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(hook))
    model.eval()
    with torch.no_grad():
        x = torch.randn(1, int(input_channels), int(input_length), device=device)
        if "cwt" in getattr(model, "enabled_views", ()):
            branch = getattr(model, "cwt_branch", None)
            conv_factor = 1 if getattr(branch, "wavelet_mode", "real_morlet") == "real_morlet" else 2
            scale_count = int(getattr(branch, "wavelet_scale_end", 64)) - int(
                getattr(branch, "wavelet_scale_start", 1)
            ) + 1
            kernel_size = int(getattr(branch, "wavelet_kernel_size", 129))
            macs["CWT_functional"] = int(conv_factor * scale_count * int(input_length) * kernel_size)
        _ = model(x)
    for item in hooks:
        item.remove()
    total = int(sum(macs.values()))
    return {
        "approx_macs": total,
        "approx_flops": total * 2,
        "approx_mflops_if_1mac_2flops": total * 2 / 1_000_000.0,
        "macs_by_op": macs,
        "note": "Conv/Linear hooks plus functional CWT estimate; called before training to preserve canonical RNG order.",
    }


def resolve_data_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (ROOT / path).resolve()


def _frame_hash(df: pd.DataFrame) -> str:
    rows = df.astype(str).agg("|".join, axis=1).tolist()
    return hashlib.sha256(("\n".join(rows)).encode("utf-8")).hexdigest()


def _manifest_class_counts(df: pd.DataFrame, label_col: str) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for split_name, group in df.groupby("split"):
        values = group[label_col].astype(int).value_counts().sort_index()
        counts[str(split_name)] = {str(k): int(v) for k, v in values.items()}
    return counts


def split_manifest(config: dict[str, Any]) -> dict[str, Any]:
    dataset = config["dataset"]
    name = str(dataset.get("name", dataset.get("dataset_name", ""))).lower().replace("_", "-")
    if dataset.get("split_dir"):
        split_dir = resolve_under_root(dataset["split_dir"])
        manifest_path = split_dir / "metadata.json"
        if manifest_path.exists():
            return json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_value = dataset.get("manifest")
    if not manifest_value:
        raise FileNotFoundError("Split manifest not found and dataset.manifest is not configured.")
    manifest_path = Path(manifest_value)
    if not manifest_path.is_absolute() and dataset.get("data_root"):
        data_candidate = Path(dataset["data_root"]) / manifest_path
        manifest_path = data_candidate if data_candidate.exists() else resolve_data_path(manifest_path)
    else:
        manifest_path = resolve_data_path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")
    df = pd.read_csv(manifest_path)
    if "split" not in df.columns:
        raise ValueError(f"{manifest_path} must contain a split column.")
    label_col = "label_id" if "label_id" in df.columns else "label"
    if label_col not in df.columns:
        raise ValueError(f"{manifest_path} must contain label or label_id.")
    split_hashes = {
        split_name: _frame_hash(df[df["split"].astype(str) == split_name].reset_index(drop=True))
        for split_name in ("train", "val", "test")
    }
    out: dict[str, Any] = {
        "dataset": dataset.get("name", dataset.get("dataset_name", name)),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "split_seed": int(dataset.get("split_seed", config.get("training", {}).get("split_seed", 42))),
        "canonical_split_hashes": split_hashes,
        "split_counts": {
            split_name: int((df["split"].astype(str) == split_name).sum())
            for split_name in ("train", "val", "test")
        },
        "class_counts": _manifest_class_counts(df, label_col),
        "label_column": label_col,
    }
    if name in {"fiberrisk", "fiber-risk", "fiberrisks"}:
        data_root = Path(dataset.get("data_root", manifest_path.parent)).resolve()
        for split_name in ("train", "val", "test"):
            for suffix in ("x", "y"):
                path = data_root / f"{split_name}_{suffix}.npy"
                out[f"{split_name}_{suffix}_sha256"] = sha256_file(path) if path.exists() else None
    return out


def _metadata_for_loader(loader) -> list[dict[str, Any]]:
    if hasattr(loader.dataset, "prediction_metadata"):
        return loader.dataset.prediction_metadata()
    df = loader.dataset.manifest.reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        rows.append(
            {
                "sample_id": str(row.get("sample_id", f"test_{idx:06d}")),
                "file_name": str(row.get("file_name", "")),
                "label_name": str(row.get("label_name", "")),
                "start": int(row.get("start", 0)),
                "end": int(row.get("end", 0)),
            }
        )
    return rows


def split_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, dict) and "x" in batch and "label" in batch:
        return batch["x"], batch["label"]
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise ValueError("Expected batch with x and label tensors.")


def forward_logits(
    model: nn.Module,
    x: torch.Tensor,
    device: torch.device,
    eval_window_batch_size: int | None = None,
) -> tuple[torch.Tensor, float, int]:
    if x.dim() == 3:
        x = x.to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        tick = time.perf_counter()
        out = model(x, return_features=True)
        logits = out["logits"] if isinstance(out, dict) else out
        if device.type == "cuda":
            torch.cuda.synchronize()
        return logits, float(time.perf_counter() - tick), int(x.shape[0])
    if x.dim() != 4:
        raise ValueError(f"Expected [B,C,L] or [B,Nw,C,L], got {tuple(x.shape)}.")
    batch_size, num_windows, channels, length = x.shape
    flat = x.reshape(batch_size * num_windows, channels, length)
    chunks: list[torch.Tensor] = []
    forward_seconds = 0.0
    chunk_size = int(eval_window_batch_size or flat.shape[0])
    for start_idx in range(0, flat.shape[0], chunk_size):
        part = flat[start_idx:start_idx + chunk_size].to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        tick = time.perf_counter()
        out = model(part, return_features=True)
        logits = out["logits"] if isinstance(out, dict) else out
        if device.type == "cuda":
            torch.cuda.synchronize()
        forward_seconds += time.perf_counter() - tick
        chunks.append(logits)
    window_logits = torch.cat(chunks, dim=0).reshape(batch_size, num_windows, -1)
    return window_logits.mean(dim=1), float(forward_seconds), int(batch_size)


@torch.no_grad()
def evaluate_model(model: nn.Module, loader, criterion: nn.Module, device: torch.device, split: str) -> tuple[dict[str, Any], np.ndarray, list[int], list[int], np.ndarray]:
    model.eval()
    losses: list[float] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    probs: list[np.ndarray] = []
    forward_seconds = 0.0
    eval_window_batch_size = int(getattr(loader.dataset, "eval_window_batch_size", 0) or 0)
    for batch in loader:
        x, y = split_batch(batch)
        y = y.to(device, non_blocking=True)
        logits, elapsed, _ = forward_logits(model, x, device, eval_window_batch_size=eval_window_batch_size)
        forward_seconds += elapsed
        losses.append(float(criterion(logits, y).item()))
        y_true.extend(y.detach().cpu().numpy().astype(np.int64).tolist())
        y_pred.extend(torch.argmax(logits, dim=1).detach().cpu().numpy().astype(np.int64).tolist())
        probs.append(softmax_probabilities(logits).detach().cpu().numpy().astype(np.float32))
    prob_all = np.concatenate(probs, axis=0) if probs else np.zeros((0, int(model.num_classes)), dtype=np.float32)
    metrics = compute_classification_metrics(y_true, y_pred, int(model.num_classes))
    matrix = metrics.pop("confusion_matrix")
    metrics["loss"] = float(np.mean(losses)) if losses else math.nan
    metrics["split"] = split
    metrics["inference_time_ms_per_sample"] = float(forward_seconds * 1000.0 / max(1, len(y_true)))
    return metrics, matrix, y_true, y_pred, prob_all


def train_one_epoch(model: nn.Module, loader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> dict[str, Any]:
    model.train()
    losses: list[float] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    start = time.perf_counter()
    for batch in loader:
        x, y = split_batch(batch)
        if x.dim() != 3:
            raise ValueError(f"Training expects one waveform per sample, got {tuple(x.shape)}.")
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        out = model(x, return_features=True)
        logits = out["logits"] if isinstance(out, dict) else out
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
        y_true.extend(y.detach().cpu().numpy().astype(np.int64).tolist())
        y_pred.extend(torch.argmax(logits, dim=1).detach().cpu().numpy().astype(np.int64).tolist())
    metrics = compute_classification_metrics(y_true, y_pred, int(model.num_classes))
    metrics.pop("confusion_matrix", None)
    metrics["loss"] = float(np.mean(losses)) if losses else math.nan
    metrics["epoch_seconds"] = float(time.perf_counter() - start)
    return metrics


class BestTracker:
    def __init__(self, patience: int) -> None:
        self.patience = int(patience)
        self.best_macro_f1 = -float("inf")
        self.best_accuracy = -float("inf")
        self.best_epoch = 0
        self.bad_epochs = 0

    def update(self, epoch: int, val_metrics: dict[str, Any]) -> bool:
        score = float(val_metrics["macro_f1"])
        accuracy = float(val_metrics["accuracy"])
        improved = score > self.best_macro_f1 or (score == self.best_macro_f1 and accuracy > self.best_accuracy)
        if improved:
            self.best_macro_f1 = score
            self.best_accuracy = accuracy
            self.best_epoch = int(epoch)
            self.bad_epochs = 0
            return True
        self.bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience


def save_test_outputs(
    result_dir: Path,
    metrics: dict[str, Any],
    matrix: np.ndarray,
    y_true: list[int],
    y_pred: list[int],
    probs: np.ndarray,
    loader,
    label_names: list[str],
    dataset_name: str,
) -> None:
    metadata = _metadata_for_loader(loader)
    save_prediction_tables(result_dir, y_true, y_pred, probs, label_names, metadata)
    save_confusion_matrix_csv(matrix, label_names, result_dir / "confusion_matrix.csv")
    save_confusion_matrix_csv(matrix, label_names, result_dir / "test_confusion_matrix.csv")
    title = f"TBAGNet {dataset_name} Test Confusion Matrix"
    plot_confusion_matrix(matrix, label_names, result_dir / "confusion_matrix.png", title)
    plot_confusion_matrix(matrix, label_names, result_dir / "test_confusion_matrix.png", title)
    save_per_class_metrics(result_dir / "per_class_metrics.csv", y_true, y_pred, label_names)
    write_json(result_dir / "inference_time.json", {"inference_time_ms_per_sample": metrics["inference_time_ms_per_sample"]})


def _prepare_output_dirs(config: dict[str, Any], force: bool) -> tuple[Path, Path, Path]:
    result_dir = Path(config["output"]["result_dir"])
    checkpoint_dir = Path(config["output"]["checkpoint_dir"])
    log_dir = Path(config["output"]["log_dir"])
    completed = result_dir / "completed.flag"
    if completed.exists() and not force:
        raise RuntimeError(f"Completed run already exists: {public_path(result_dir)}. Use --force only after audit.")
    if result_dir.exists() and any(result_dir.iterdir()) and not force:
        raise RuntimeError(f"Non-empty result directory exists: {public_path(result_dir)}. Use --force only after audit.")
    if force:
        for target in [result_dir, checkpoint_dir, log_dir]:
            resolved = target.resolve()
            if resolved != ROOT.resolve() and ROOT.resolve() in resolved.parents:
                shutil.rmtree(resolved, ignore_errors=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    return result_dir, checkpoint_dir, log_dir


def run_training(config: dict[str, Any], data_root: str | Path | None = None, output_dir: str | Path | None = None, force: bool = False, command: str | None = None) -> dict[str, Any]:
    config = resolve_config(config, data_root=data_root, output_dir=output_dir)
    result_dir, checkpoint_dir, log_dir = _prepare_output_dirs(config, force=force)
    setup_logging(log_dir)
    run_id = (
        f"{config['dataset']['name']}_{config['experiment']['name']}_seed{config['training']['seed']}_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    config["run_id"] = run_id

    requested_device = str(config.get("device", "cpu"))
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {requested_device}, but CUDA is unavailable.")
    device = torch.device(requested_device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    cfg_for_save = redacted_config(config)
    write_yaml(result_dir / "full_config.yaml", cfg_for_save)
    write_yaml(result_dir / "config.yaml", cfg_for_save)
    (result_dir / "command.txt").write_text(redact_local_paths(command or " ".join(sys.argv)), encoding="utf-8")
    write_json(result_dir / "source_hashes.json", source_hashes())
    write_json(result_dir / "environment.json", environment_report())
    split_info = split_manifest(config)
    write_json(result_dir / "split_manifest.json", split_info)

    training = config["training"]
    seed_info = seed_everything(int(training["seed"]), deterministic=bool(training["deterministic"]))
    torch.backends.cudnn.benchmark = bool(training["cudnn_benchmark"])
    torch.backends.cudnn.deterministic = bool(training["cudnn_deterministic"])
    determinism = {
        **seed_info,
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
    }
    write_json(result_dir / "determinism.json", determinism)

    loaders = build_dataloaders(config, data_root=config["dataset"].get("data_root"))
    label_names = read_label_names(config["dataset"], int(config["dataset"]["num_classes"]))
    model = build_model(config).to(device)
    initial_model_hash = state_dict_hash(model)
    criterion = build_criterion(config, device)
    optimizer = build_optimizer(config, model)
    complexity = estimate_macs(
        model,
        device,
        input_length=int(config["dataset"]["input_length"]),
        input_channels=int(config["dataset"]["input_channels"]),
    )
    params = count_params(model)
    complexity.update({"parameter_count": params, "params_m": params / 1_000_000.0})
    write_json(result_dir / "parameter_flops_latency.json", {**complexity, "flops": int(complexity["approx_flops"])})

    logging.info(
        "experiment_config wavelet_mode=%s run_seed=%s split_seed=%s split_hash=%s "
        "kernel_mean_subtraction=%s pooling_mode=%s deterministic=%s cudnn_benchmark=%s "
        "spectral_relation_depth=%s tfpatch_relation_depth=%s branch_set=%s fusion_type=%s model_parameters=%d",
        config["model"]["wavelet_mode"],
        training["run_seed"],
        training["split_seed"],
        json.dumps(split_info["canonical_split_hashes"], sort_keys=True),
        config["model"]["kernel_mean_subtraction"],
        config["model"]["pooling_mode"],
        training["deterministic"],
        training["cudnn_benchmark"],
        config["model"]["spectral_relation_depth"],
        config["model"]["tfpatch_relation_depth"],
        config["model"]["branches"],
        config["model"]["fusion_type"],
        params,
    )
    logging.info("best checkpoint will be selected by validation Macro-F1, then validation Accuracy on exact ties")
    logging.info("dataset_root=%s", config["dataset"].get("data_root", ""))
    logging.info("dataset_manifest=%s", config["dataset"].get("manifest", config["dataset"].get("split_dir", "")))
    logging.info("rng_protocol=%s", config["rng_protocol"])
    logging.info("determinism=%s", json.dumps(determinism, sort_keys=True))
    logging.info("initial_model_state_sha256=%s", initial_model_hash)
    logging.info("train/val/test samples=%d/%d/%d", len(loaders["train"].dataset), len(loaders["val"].dataset), len(loaders["test"].dataset))
    skipped_rows: list[dict[str, Any]] = []
    for split_name, loader in loaders.items():
        for item in getattr(loader.dataset, "skipped_records", []):
            skipped_rows.append({"loader_split": split_name, **item})
    if skipped_rows:
        pd.DataFrame(skipped_rows).to_csv(result_dir / "skipped_files.csv", index=False, encoding="utf-8-sig")
        logging.warning("skipped_files=%d details=%s", len(skipped_rows), public_path(result_dir / "skipped_files.csv"))
    else:
        pd.DataFrame(columns=["loader_split", "split", "sample_id", "class_name", "file_path", "reason"]).to_csv(
            result_dir / "skipped_files.csv", index=False, encoding="utf-8-sig"
        )

    history: list[dict[str, Any]] = []
    tracker = BestTracker(int(training["patience"]))
    best_path = checkpoint_dir / "best_model.pth"
    training_start = datetime.now(timezone.utc).isoformat()
    start_train = time.perf_counter()
    best_payload: dict[str, Any] | None = None

    for epoch in range(1, int(training["max_epochs"]) + 1):
        if hasattr(loaders["train"].dataset, "set_epoch"):
            loaders["train"].dataset.set_epoch(epoch)
        train_metrics = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        val_metrics, _, _, _, _ = evaluate_model(model, loaders["val"], criterion, device, "val")
        row = {"epoch": epoch}
        row.update({f"train_{k}": v for k, v in train_metrics.items() if isinstance(v, (int, float, str))})
        row.update({f"val_{k}": v for k, v in val_metrics.items() if isinstance(v, (int, float, str))})
        history.append(row)
        pd.DataFrame(history).to_csv(result_dir / "epoch_metrics.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(history).to_csv(result_dir / "metrics.csv", index=False, encoding="utf-8-sig")
        payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": None,
            "scaler_state": None,
            "config": cfg_for_save,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "run_id": run_id,
            "initial_model_state_sha256": initial_model_hash,
        }
        save_checkpoint(checkpoint_dir / "last_model.pth", payload)
        improved = tracker.update(epoch, val_metrics)
        logging.info(
            "epoch=%d train_acc=%.6f train_macro_f1=%.6f val_acc=%.6f val_macro_f1=%.6f seconds=%.1f",
            epoch,
            train_metrics["accuracy"],
            train_metrics["macro_f1"],
            val_metrics["accuracy"],
            val_metrics["macro_f1"],
            train_metrics["epoch_seconds"],
        )
        if improved:
            best_payload = dict(payload)
            best_payload["best_epoch"] = tracker.best_epoch
            best_payload["best_val_macro_f1"] = tracker.best_macro_f1
            best_payload["best_val_accuracy"] = tracker.best_accuracy
            save_checkpoint(best_path, best_payload)
            logging.info("updated_best epoch=%d val_macro_f1=%.6f val_acc=%.6f path=%s", epoch, tracker.best_macro_f1, tracker.best_accuracy, public_path(best_path))
        if tracker.should_stop:
            logging.info("early_stop epoch=%d best_epoch=%d best_val_macro_f1=%.6f", epoch, tracker.best_epoch, tracker.best_macro_f1)
            break

    if best_payload is None or not best_path.exists():
        raise RuntimeError("No best checkpoint was saved.")
    checkpoint = load_checkpoint_payload(best_path, map_location=device)
    model.load_state_dict(checkpoint_state_dict(checkpoint), strict=True)
    val_metrics, _, _, _, _ = evaluate_model(model, loaders["val"], criterion, device, "val")
    test_metrics, matrix, y_true, y_pred, probs = evaluate_model(model, loaders["test"], criterion, device, "test")
    save_test_outputs(
        result_dir,
        test_metrics,
        matrix,
        y_true,
        y_pred,
        probs,
        loaders["test"],
        label_names,
        str(config["dataset"]["name"]),
    )

    training_time = time.perf_counter() - start_train
    metrics = {
        "run_id": run_id,
        "run_status": "completed",
        "dataset": config["dataset"]["name"],
        "branches": config["model"]["branches"],
        "branch_set": config["model"]["branch_set"],
        "fusion_type": model.fusion_type,
        "seed": int(training["seed"]),
        "run_seed": int(training["run_seed"]),
        "split_seed": int(training["split_seed"]),
        "split_hashes": split_info["canonical_split_hashes"],
        "wavelet_mode": config["model"]["wavelet_mode"],
        "wavelet_subbranch_mode": config["model"]["wavelet_subbranch_mode"],
        "kernel_mean_subtraction": bool(config["model"]["kernel_mean_subtraction"]),
        "pooling_mode": config["model"]["pooling_mode"],
        "spectral_relation_depth": int(config["model"]["spectral_relation_depth"]),
        "tfpatch_relation_depth": int(config["model"]["tfpatch_relation_depth"]),
        "best_epoch": int(tracker.best_epoch),
        "best_val_metric": float(tracker.best_macro_f1),
        "best_validation_accuracy": float(val_metrics["accuracy"]),
        "best_validation_macro_precision": float(val_metrics["macro_precision"]),
        "best_validation_macro_recall": float(val_metrics["macro_recall"]),
        "best_validation_macro_f1": float(val_metrics["macro_f1"]),
        "best_validation_weighted_f1": float(val_metrics["weighted_f1"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_macro_precision": float(test_metrics["macro_precision"]),
        "test_macro_recall": float(test_metrics["macro_recall"]),
        "test_macro_f1": float(test_metrics["macro_f1"]),
        "test_weighted_precision": float(test_metrics["weighted_precision"]),
        "test_weighted_recall": float(test_metrics["weighted_recall"]),
        "test_weighted_f1": float(test_metrics["weighted_f1"]),
        "accuracy": float(test_metrics["accuracy"]),
        "macro_precision": float(test_metrics["macro_precision"]),
        "macro_recall": float(test_metrics["macro_recall"]),
        "macro_f1": float(test_metrics["macro_f1"]),
        "weighted_precision": float(test_metrics["weighted_precision"]),
        "weighted_recall": float(test_metrics["weighted_recall"]),
        "weighted_f1": float(test_metrics["weighted_f1"]),
        "test_loss": float(test_metrics["loss"]),
        "parameter_count": params,
        "total_parameters": params,
        "trainable_parameters": params,
        "approx_macs": int(complexity["approx_macs"]),
        "approx_flops": int(complexity["approx_flops"]),
        "approx_mflops_if_1mac_2flops": float(complexity["approx_mflops_if_1mac_2flops"]),
        "train_samples": len(loaders["train"].dataset),
        "validation_samples": len(loaders["val"].dataset),
        "test_samples": len(loaders["test"].dataset),
        "test_predictions": len(y_pred),
        "metric_unit": "original_mat_sample" if str(config["dataset"]["name"]).lower() == "bjdas" else "manifest_row",
        "training_start_time": training_start,
        "training_end_time": datetime.now(timezone.utc).isoformat(),
        "training_time_seconds": float(training_time),
        "epochs_completed": int(len(history)),
        "early_stopped": bool(tracker.should_stop),
        "inference_time_ms_per_sample": float(test_metrics["inference_time_ms_per_sample"]),
        "checkpoint_path": public_path(best_path),
        "best_model_checkpoint_path": public_path(best_path),
        "best_checkpoint_sha256": sha256_file(best_path),
        "result_dir": public_path(result_dir),
        "initial_model_state_sha256": initial_model_hash,
        "state_dict_key_count": len(checkpoint_state_dict(checkpoint)),
        "state_dict_numel": int(sum(v.numel() for v in checkpoint_state_dict(checkpoint).values() if hasattr(v, "numel"))),
        "prediction_hash": prediction_hash(y_true, y_pred, probs),
        "confusion_matrix": matrix.tolist(),
        "determinism": determinism,
        "rng_protocol": config["rng_protocol"],
        "source_hashes": source_hashes(),
    }
    write_json(result_dir / "best_metrics.json", metrics)
    write_json(result_dir / "metrics.json", metrics)
    write_json(
        result_dir / "parameter_flops_latency.json",
        {**complexity, "flops": int(complexity["approx_flops"]), "inference_time_ms_per_sample": metrics["inference_time_ms_per_sample"]},
    )
    (result_dir / "completed.flag").write_text(f"completed_at_utc={datetime.now(timezone.utc).isoformat()}\nrun_id={run_id}\n", encoding="utf-8")
    logging.info("completed %s", json.dumps({k: metrics[k] for k in ["test_accuracy", "test_macro_f1", "test_weighted_f1", "best_epoch"]}, sort_keys=True))
    return metrics


def evaluate_checkpoint(config: dict[str, Any], checkpoint_path: str | Path, data_root: str | Path, output_dir: str | Path, device: str | None = None) -> dict[str, Any]:
    cfg = resolve_config(config, data_root=data_root, output_dir=output_dir)
    if device is not None:
        cfg["device"] = device
    result_dir = Path(cfg["output"]["result_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)
    requested_device = torch.device(cfg.get("device", "cpu"))
    if requested_device.type == "cuda":
        torch.cuda.set_device(0 if requested_device.index is None else requested_device.index)
    seed_everything(int(cfg["training"]["seed"]), deterministic=bool(cfg["training"]["deterministic"]))
    loaders = build_dataloaders(cfg, data_root=cfg["dataset"].get("data_root"))
    payload = load_checkpoint_payload(checkpoint_path, map_location=requested_device)
    model = build_model(cfg).to(requested_device)
    try:
        model.load_state_dict(checkpoint_state_dict(payload), strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Checkpoint state_dict is incompatible with the model defined by YAML config: {checkpoint_path}\n"
            f"Strict loading details:\n{exc}"
        ) from exc
    criterion = build_criterion(cfg, requested_device)
    label_names = read_label_names(cfg["dataset"], int(cfg["dataset"]["num_classes"]))
    test_metrics, matrix, y_true, y_pred, probs = evaluate_model(model, loaders["test"], criterion, requested_device, "test")
    save_test_outputs(
        result_dir,
        test_metrics,
        matrix,
        y_true,
        y_pred,
        probs,
        loaders["test"],
        label_names,
        str(cfg["dataset"]["name"]),
    )
    metrics = {
        "checkpoint_path": public_path(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_macro_precision": float(test_metrics["macro_precision"]),
        "test_macro_recall": float(test_metrics["macro_recall"]),
        "test_macro_f1": float(test_metrics["macro_f1"]),
        "test_weighted_precision": float(test_metrics["weighted_precision"]),
        "test_weighted_recall": float(test_metrics["weighted_recall"]),
        "test_weighted_f1": float(test_metrics["weighted_f1"]),
        "inference_time_ms_per_sample": float(test_metrics["inference_time_ms_per_sample"]),
        "prediction_hash": prediction_hash(y_true, y_pred, probs),
        "parameter_count": count_params(model),
        "split_hashes": split_manifest(cfg)["canonical_split_hashes"],
    }
    write_json(result_dir / "checkpoint_reevaluation.json", metrics)
    return metrics
