from __future__ import annotations

import csv
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset

from tbagnet.utils.seed import loader_generator, seed_worker
from .preprocessing import sample_zscore


RELEASE_ROOT = Path(__file__).resolve().parents[1]


def _resolve_path(path: str | Path, base: str | Path | None = None) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path.resolve()
    if base is not None:
        return (Path(base) / path).resolve()
    return (RELEASE_ROOT / path).resolve()


def _dataset_name(config: dict[str, Any]) -> str:
    return str(config.get("name", config.get("dataset_name", ""))).lower().replace("_", "-")


def _repair_utf8_gbk_mojibake(path: Path) -> Path:
    repaired_parts: list[str] = []
    changed = False
    for part in path.parts:
        try:
            repaired = part.encode("gbk").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            repaired = part
        repaired_parts.append(repaired)
        changed = changed or repaired != part
    return Path(*repaired_parts) if changed else path


class RailwayDASSplitDataset(Dataset):
    REQUIRED_COLUMNS = {"relative_path", "label", "label_name", "start", "end", "split"}

    def __init__(
        self,
        split_csv: str | Path,
        data_root: str | Path,
        input_length: int = 1024,
        data_key: str = "signal",
        normalize: bool = True,
        mat_cache_size: int = 8,
    ):
        self.split_csv = Path(split_csv).resolve()
        self.data_root = Path(data_root).resolve()
        self.input_length = int(input_length)
        self.data_key = str(data_key)
        self.normalize = bool(normalize)
        self.mat_cache_size = int(mat_cache_size)
        self._signal_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        df = pd.read_csv(self.split_csv)
        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"{self.split_csv} missing columns: {sorted(missing)}")
        self.manifest = df.reset_index(drop=True)
        self.split = str(self.manifest["split"].iloc[0]) if len(self.manifest) else ""
        self.labels = self.manifest["label"].astype(np.int64).to_numpy()
        self.skipped_records: list[dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self.manifest)

    def _mat_path(self, row) -> Path:
        return (self.data_root / Path(str(row["relative_path"]))).resolve()

    def _read_signal(self, path: Path) -> np.ndarray:
        key = str(path)
        if self.mat_cache_size > 0 and key in self._signal_cache:
            signal = self._signal_cache.pop(key)
            self._signal_cache[key] = signal
            return signal
        mat = loadmat(path)
        if self.data_key in mat:
            signal = mat[self.data_key]
        else:
            numeric = [v for k, v in mat.items() if not k.startswith("__") and np.issubdtype(np.asarray(v).dtype, np.number)]
            if not numeric:
                raise ValueError(f"No numeric signal found in {path}")
            signal = numeric[0]
        signal = np.asarray(signal)
        if np.iscomplexobj(signal):
            signal = np.real(signal)
        signal = np.squeeze(signal).astype(np.float32, copy=False)
        if signal.ndim != 1:
            raise ValueError(f"Expected 1D signal in {path}, got {signal.shape}.")
        if self.mat_cache_size > 0:
            self._signal_cache[key] = signal
            while len(self._signal_cache) > self.mat_cache_size:
                self._signal_cache.popitem(last=False)
        return signal

    def __getitem__(self, index: int):
        row = self.manifest.iloc[index]
        start = int(row["start"])
        end = int(row["end"])
        if end - start != self.input_length:
            raise ValueError(f"Window length {end - start} != input_length={self.input_length}")
        path = self._mat_path(row)
        if not path.exists():
            raise FileNotFoundError(f"MAT file not found: {path}")
        window = self._read_signal(path)[start:end].astype(np.float32, copy=True)
        if window.shape[0] != self.input_length:
            raise ValueError(f"Window length from {path} is {window.shape[0]}, expected {self.input_length}")
        if self.normalize:
            window = sample_zscore(window)
        return torch.from_numpy(window.reshape(1, self.input_length)), torch.tensor(int(row["label"]), dtype=torch.long)

    def prediction_metadata(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for idx, row in self.manifest.reset_index(drop=True).iterrows():
            rows.append(
                {
                    "sample_id": str(row.get("sample_id", f"{self.split}_{idx:06d}")),
                    "file_name": str(row.get("relative_path", row.get("file_name", ""))),
                    "label_name": str(row.get("label_name", "")),
                    "start": int(row.get("start", 0)),
                    "end": int(row.get("end", 0)),
                }
            )
        return rows


class RailwayManifestDataset(RailwayDASSplitDataset):
    """Compatibility wrapper for baseline code that expects a manifest CSV."""

    def __init__(
        self,
        manifest_csv: str | Path,
        split: str,
        input_length: int = 1024,
        data_key: str = "signal",
        normalize: bool = True,
        mat_cache_size: int = 8,
    ) -> None:
        manifest_csv = Path(manifest_csv)
        split_csv = manifest_csv.parent / f"{split}.csv" if manifest_csv.name == "manifest.csv" else manifest_csv
        data_root = manifest_csv.parent
        super().__init__(
            split_csv=split_csv,
            data_root=data_root,
            input_length=input_length,
            data_key=data_key,
            normalize=normalize,
            mat_cache_size=mat_cache_size,
        )


class EfficientDVSDataset(Dataset):
    """Efficient-DVS per-file NPY dataset.

    Split CSV columns are ``filepath``, ``label``, ``class_name``, and
    ``split``. Samples are stored as [length, channels] and returned as
    [channels, length].
    """

    REQUIRED_COLUMNS = {"filepath", "label", "class_name", "split"}

    def __init__(
        self,
        csv_path: str | Path,
        data_root: str | Path,
        expected_shape: tuple[int, int] = (256, 11),
        normalize: str = "none",
    ) -> None:
        self.csv_path = Path(csv_path).resolve()
        self.data_root = Path(data_root).resolve()
        self.expected_shape = tuple(int(v) for v in expected_shape)
        self.normalize = str(normalize)
        if self.normalize not in {"none", "sample_zscore"}:
            raise ValueError("normalize must be none or sample_zscore.")
        with self.csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            missing = self.REQUIRED_COLUMNS - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{self.csv_path} is missing columns: {sorted(missing)}")
            self.records = [dict(row) for row in reader]
        self.labels = np.asarray([int(row["label"]) for row in self.records], dtype=np.int64)
        self.manifest = pd.DataFrame(self.records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        raw_path = Path(record["filepath"])
        path = raw_path if raw_path.is_absolute() else self.data_root / raw_path
        if not path.exists():
            repaired_path = _repair_utf8_gbk_mojibake(path)
            if repaired_path.exists():
                path = repaired_path
        array = np.load(path, allow_pickle=False)
        if tuple(array.shape) != self.expected_shape:
            raise ValueError(f"Expected {self.expected_shape} for {path}, got {tuple(array.shape)}.")
        x = np.asarray(array, dtype=np.float32)
        if self.normalize == "sample_zscore":
            mean = float(x.mean())
            std = max(float(x.std()), 1e-5)
            x = ((x - mean) / std).astype(np.float32, copy=False)
        return torch.from_numpy(np.ascontiguousarray(x.T)), torch.tensor(int(record["label"]), dtype=torch.long)


class NPYWaveformDataset(Dataset):
    """Prepared waveform dataset with {split}_x.npy and {split}_y.npy."""

    def __init__(
        self,
        data_root: str | Path,
        split: str,
        input_length: int | None = None,
        normalize: bool = True,
        manifest_csv: str | Path | None = None,
    ) -> None:
        self.data_root = Path(data_root).resolve()
        self.split = str(split)
        self.normalize = bool(normalize)
        self.input_length = None if input_length is None else int(input_length)
        self.skipped_records: list[dict[str, Any]] = []
        x_path = self.data_root / f"{self.split}_x.npy"
        y_path = self.data_root / f"{self.split}_y.npy"
        if not x_path.exists() or not y_path.exists():
            raise FileNotFoundError(f"Missing split arrays: {x_path} and {y_path}")
        self.x_path = x_path
        self.y_path = y_path
        self.x = np.load(x_path, mmap_mode="r")
        self.y = np.load(y_path, mmap_mode="r").astype(np.int64)
        if self.x.ndim == 1:
            if self.input_length is None:
                raise ValueError(f"{x_path} is flat; input_length is required.")
            self.x = self.x.reshape(-1, self.input_length)
        if self.x.ndim != 2:
            raise ValueError(f"Expected [N, T] waveform array, got {self.x.shape}.")
        if self.input_length is not None and self.x.shape[1] != self.input_length:
            raise ValueError(f"Expected input_length={self.input_length}, got {self.x.shape[1]}.")
        if self.x.shape[0] != self.y.shape[0]:
            raise ValueError(f"X/y length mismatch: {self.x.shape[0]} vs {self.y.shape[0]}.")
        self.labels = np.asarray(self.y, dtype=np.int64)
        self.manifest: pd.DataFrame | None = None
        if manifest_csv is not None:
            manifest_path = Path(manifest_csv).resolve()
            if manifest_path.exists():
                manifest = pd.read_csv(manifest_path)
                if "split" in manifest.columns:
                    manifest = manifest[manifest["split"].astype(str) == self.split].reset_index(drop=True)
                if len(manifest) == len(self.y):
                    self.manifest = manifest

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, index: int):
        window = np.asarray(self.x[index], dtype=np.float32).copy()
        if self.normalize:
            window = sample_zscore(window)
        x = window.reshape(1, window.shape[0]).astype(np.float32, copy=False)
        return torch.from_numpy(x), torch.tensor(int(self.y[index]), dtype=torch.long)

    def prediction_metadata(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for idx in range(len(self)):
            if self.manifest is not None:
                row = self.manifest.iloc[idx]
                label_name = str(row.get("label_name", ""))
                sample_id = str(row.get("sample_id", row.get("mat_path", f"{self.split}:{idx}")))
                file_name = str(row.get("mat_path", ""))
                start = int(row.get("start", 0))
                end = int(row.get("end", self.input_length or 0))
            else:
                label_name = ""
                sample_id = f"{self.split}:{idx}"
                file_name = str(self.x_path.name)
                start = 0
                end = int(self.input_length or self.x.shape[1])
            rows.append(
                {
                    "sample_id": sample_id,
                    "file_name": file_name,
                    "label_name": label_name,
                    "start": start,
                    "end": end,
                }
            )
        return rows


def deterministic_starts(length: int, window_size: int, stride: int) -> list[int]:
    if length <= window_size:
        return [0]
    starts = list(range(0, length - window_size + 1, stride))
    tail = length - window_size
    if starts[-1] != tail:
        starts.append(tail)
    return starts


def reduce_spatial_rms(array: np.ndarray, expected_shape: tuple[int, int] = (10000, 12)) -> np.ndarray:
    raw = np.asarray(array)
    if raw.shape != expected_shape:
        raise ValueError(f"BJDAS expected raw shape {expected_shape}, got {tuple(raw.shape)}.")
    if not np.issubdtype(raw.dtype, np.number):
        raise ValueError(f"BJDAS signal must be numeric, got dtype={raw.dtype}.")
    if not np.isfinite(raw).all():
        raise ValueError("BJDAS raw signal contains NaN or Inf.")
    values = raw.astype(np.float32, copy=False)
    signal = np.sqrt(np.mean(np.square(values), axis=1, dtype=np.float32))
    if signal.shape != (expected_shape[0],):
        raise RuntimeError(f"BJDAS RMS reduction produced {signal.shape}, expected {(expected_shape[0],)}.")
    if not np.isfinite(signal).all():
        raise ValueError("BJDAS RMS-reduced signal contains NaN or Inf.")
    return np.ascontiguousarray(signal, dtype=np.float32)


class BJDASMatDataset(Dataset):
    """BJDAS original-MAT dataset.

    Train split returns one seeded random [1, L] crop per original MAT per epoch.
    Val/test splits return all deterministic windows as [num_windows, 1, L].
    """

    REQUIRED_COLUMNS = {"file_path", "split", "class_name", "label_id", "sample_id", "mat_key", "original_shape"}

    def __init__(
        self,
        manifest_csv: str | Path,
        split: str,
        input_length: int = 1024,
        stride: int = 512,
        mat_key: str = "data",
        spatial_reduce: str = "rms",
        normalize: str = "sample_zscore",
        seed: int = 42,
        expected_shape: tuple[int, int] = (10000, 12),
        data_root: str | Path | None = None,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError("BJDAS split must be train, val, or test.")
        if mat_key != "data":
            raise ValueError("BJDAS requires MAT key `data`.")
        if spatial_reduce != "rms":
            raise ValueError("BJDAS requires spatial_reduce=rms.")
        if normalize != "sample_zscore":
            raise ValueError("BJDAS requires normalize=sample_zscore.")
        self.manifest_csv = Path(manifest_csv).resolve()
        self.data_root = None if data_root is None else Path(data_root).resolve()
        manifest = pd.read_csv(self.manifest_csv)
        missing = self.REQUIRED_COLUMNS - set(manifest.columns)
        if missing:
            raise ValueError(f"BJDAS manifest missing columns: {sorted(missing)}")
        records = manifest[manifest["split"].astype(str) == split].reset_index(drop=True)
        if records.empty:
            raise ValueError(f"No BJDAS rows for split={split} in {self.manifest_csv}.")
        self.split = split
        self.input_length = int(input_length)
        self.stride = int(stride)
        self.mat_key = mat_key
        self.spatial_reduce = spatial_reduce
        self.normalize = normalize
        self.seed = int(seed)
        self.expected_shape = tuple(int(v) for v in expected_shape)
        self.epoch = 0
        self.skipped_records: list[dict[str, Any]] = []
        keep_rows: list[pd.Series] = []
        for _, row in records.iterrows():
            path = self._path_from_row(row)
            if not path.exists():
                self.skipped_records.append(self._skip_record(row, path, "missing_file"))
                continue
            if path.stat().st_size <= 0:
                self.skipped_records.append(self._skip_record(row, path, "empty_file"))
                continue
            keep_rows.append(row)
        if not keep_rows:
            raise ValueError(f"All BJDAS rows were skipped for split={split}.")
        self.records = pd.DataFrame(keep_rows).reset_index(drop=True)
        self.labels = self.records["label_id"].astype(np.int64).to_numpy()

    def _skip_record(self, row: pd.Series, path: Path, reason: str) -> dict[str, Any]:
        return {
            "split": self.split,
            "sample_id": str(row.get("sample_id", "")),
            "class_name": str(row.get("class_name", row.get("label_name", ""))),
            "file_path": str(path),
            "reason": reason,
        }

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _path_from_row(self, row: pd.Series) -> Path:
        value = str(row["file_path"])
        path = Path(value)
        if path.is_absolute():
            return path.resolve()
        if self.data_root is not None:
            return (self.data_root / path).resolve()
        return (self.manifest_csv.parent / path).resolve()

    def load_raw(self, index: int) -> np.ndarray:
        row = self.records.iloc[index]
        path = self._path_from_row(row)
        if not path.exists() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"BJDAS MAT is missing or empty: {path}")
        payload = loadmat(path)
        if self.mat_key not in payload:
            raise KeyError(f"BJDAS MAT key `{self.mat_key}` missing from {path}.")
        raw = np.asarray(payload[self.mat_key])
        expected_from_manifest = tuple(json.loads(str(row["original_shape"])))
        if expected_from_manifest != self.expected_shape:
            raise ValueError(
                f"BJDAS manifest shape {expected_from_manifest} does not match configured {self.expected_shape}."
            )
        if tuple(raw.shape) != self.expected_shape:
            raise ValueError(f"BJDAS expected {self.expected_shape} in {path}, got {tuple(raw.shape)}.")
        return raw

    def load_signal(self, index: int) -> np.ndarray:
        return reduce_spatial_rms(self.load_raw(index), expected_shape=self.expected_shape)

    def _window_or_pad(self, signal: np.ndarray, start: int) -> tuple[np.ndarray, bool]:
        if signal.shape[0] >= self.input_length:
            return signal[start:start + self.input_length].astype(np.float32, copy=True), False
        window = np.zeros(self.input_length, dtype=np.float32)
        window[: signal.shape[0]] = signal
        return window, True

    def starts_for_length(self, length: int) -> list[int]:
        return deterministic_starts(length, self.input_length, self.stride)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records.iloc[index]
        signal = self.load_signal(index)
        label = int(row["label_id"])
        sample_id = str(row["sample_id"])
        source_file = str(self._path_from_row(row))
        class_name = str(row["class_name"])
        if self.split == "train":
            max_start = max(0, signal.shape[0] - self.input_length)
            rng = np.random.default_rng(self.seed + self.epoch * len(self) + index)
            start = int(rng.integers(0, max_start + 1)) if max_start else 0
            window, padded = self._window_or_pad(signal, start)
            window = sample_zscore(window)
            x = torch.from_numpy(window.reshape(1, self.input_length))
            window_starts = [start]
            window_indices = [0]
        else:
            window_starts = self.starts_for_length(signal.shape[0])
            windows: list[np.ndarray] = []
            padded_flags: list[bool] = []
            for start in window_starts:
                window, padded = self._window_or_pad(signal, start)
                windows.append(window)
                padded_flags.append(padded)
            x = torch.from_numpy(sample_zscore(np.stack(windows, axis=0))[:, None, :])
            padded = any(padded_flags)
            window_indices = list(range(len(window_starts)))
        return {
            "x": x,
            "label": torch.tensor(label, dtype=torch.long),
            "sample_id": sample_id,
            "source_sample_id": sample_id,
            "original_mat_path": source_file,
            "source_file": source_file,
            "class_name": class_name,
            "window_index": torch.tensor(window_indices, dtype=torch.long),
            "window_start": torch.tensor(window_starts, dtype=torch.long),
            "num_windows": torch.tensor(len(window_starts), dtype=torch.long),
            "padded": torch.tensor(bool(padded), dtype=torch.bool),
            "original_shape": str(row["original_shape"]),
        }

    def prediction_metadata(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for idx, row in self.records.reset_index(drop=True).iterrows():
            shape = tuple(json.loads(str(row["original_shape"])))
            rows.append(
                {
                    "sample_id": str(row.get("sample_id", f"{self.split}_{idx:06d}")),
                    "file_name": str(self._path_from_row(row)),
                    "label_name": str(row.get("label_name", row.get("class_name", ""))),
                    "start": 0,
                    "end": int(shape[0]),
                    "num_windows": len(self.starts_for_length(int(shape[0]))),
                }
            )
        return rows

    def window_statistics(self) -> dict[str, Any]:
        if self.split == "train":
            return {
                "original_samples": len(self),
                "windows_per_epoch": len(self),
                "windows_per_mat_policy": "one_seeded_random_crop_per_mat_per_epoch",
                "min_windows_per_mat": 1,
                "max_windows_per_mat": 1,
            }
        counts = []
        for index in range(len(self.records)):
            shape = tuple(json.loads(str(self.records.iloc[index]["original_shape"])))
            counts.append(len(self.starts_for_length(int(shape[0]))))
        return {
            "original_samples": len(self),
            "total_deterministic_windows": int(sum(counts)),
            "min_windows_per_mat": int(min(counts)),
            "max_windows_per_mat": int(max(counts)),
            "unique_windows_per_mat": sorted(set(int(v) for v in counts)),
            "window_size": self.input_length,
            "stride": self.stride,
            "aggregation": "mean_logits",
        }


def _make_loader(dataset: Dataset, batch_size: int, shuffle: bool, training: dict[str, Any], seed: int, offset: int) -> DataLoader:
    num_workers = int(training.get("num_workers", 0))
    pin_memory = bool(training.get("pin_memory", False))
    persistent_workers = bool(training.get("persistent_workers", False)) and num_workers > 0
    drop_last = bool(training.get("drop_last", False)) if shuffle else False
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
        worker_init_fn=seed_worker,
        generator=loader_generator(seed, offset),
    )


def build_dataloaders(config: dict[str, Any], data_root: str | Path | None = None) -> dict[str, DataLoader]:
    dataset = config["dataset"]
    training = config.get("training", {})
    name = _dataset_name(dataset)
    root = Path(data_root or dataset.get("data_root", ".")).resolve()
    batch_size = int(training.get("batch_size", 64))
    seed = int(training.get("run_seed", training.get("seed", 42)))
    if name in {"railway-das", "railway das"}:
        split_dir = Path(dataset.get("split_dir", "data/splits/railway_das"))
        if not split_dir.is_absolute():
            split_dir = RELEASE_ROOT / split_dir
        common = dict(
            data_root=root,
            input_length=int(dataset.get("input_length", 1024)),
            data_key=dataset.get("data_key", "signal"),
            normalize=bool(dataset.get("normalize", True)),
            mat_cache_size=int(dataset.get("mat_cache_size", 8)),
        )
        train = RailwayDASSplitDataset(split_dir / "train.csv", **common)
        val = RailwayDASSplitDataset(split_dir / "val.csv", **common)
        test = RailwayDASSplitDataset(split_dir / "test.csv", **common)
        return {
            "train": _make_loader(train, batch_size, True, training, seed, 0),
            "val": _make_loader(val, batch_size, False, training, seed, 1),
            "test": _make_loader(test, batch_size, False, training, seed, 2),
        }
    if name in {"efficient-dvs", "efficient dvs"}:
        expected_shape = tuple(int(v) for v in dataset.get("expected_shape", [256, 11]))
        normalize = str(dataset.get("normalize", "none"))
        def split_csv_path(split: str) -> Path:
            split_ref = str(dataset.get(f"{split}_split", f"splits/{split}.csv"))
            release_candidate = _resolve_path(split_ref, base=RELEASE_ROOT)
            return release_candidate if release_candidate.exists() else _resolve_path(split_ref, base=root)

        datasets = {
            split: EfficientDVSDataset(
                csv_path=split_csv_path(split),
                data_root=root,
                expected_shape=expected_shape,
                normalize=normalize,
            )
            for split in ("train", "val", "test")
        }
        return {
            "train": _make_loader(datasets["train"], batch_size, True, training, seed, 0),
            "val": _make_loader(datasets["val"], batch_size, False, training, seed, 1),
            "test": _make_loader(datasets["test"], batch_size, False, training, seed, 2),
        }
    if name in {"fiberrisk", "fiberrisks", "fiber-risk", "metro-data", "metro data", "metro"}:
        manifest = dataset.get("manifest")
        manifest_path = None
        if manifest:
            release_candidate = _resolve_path(manifest, base=RELEASE_ROOT)
            manifest_path = release_candidate if release_candidate.exists() else _resolve_path(manifest, base=root)
        common = dict(
            data_root=root,
            input_length=int(dataset.get("input_length", 2500)),
            normalize=bool(dataset.get("normalize", True)),
            manifest_csv=manifest_path,
        )
        train = NPYWaveformDataset(split="train", **common)
        val = NPYWaveformDataset(split="val", **common)
        test = NPYWaveformDataset(split="test", **common)
        return {
            "train": _make_loader(train, batch_size, True, training, seed, 0),
            "val": _make_loader(val, batch_size, False, training, seed, 1),
            "test": _make_loader(test, batch_size, False, training, seed, 2),
        }
    if name == "bjdas":
        manifest = dataset.get("manifest")
        if not manifest:
            raise ValueError("BJDAS requires dataset.manifest.")
        manifest_path = _resolve_path(manifest, base=RELEASE_ROOT)
        common = dict(
            manifest_csv=manifest_path,
            input_length=int(dataset.get("input_length", 1024)),
            stride=int(dataset.get("stride", 512)),
            mat_key=str(dataset.get("mat_key", dataset.get("data_key", "data"))),
            spatial_reduce=str(dataset.get("spatial_reduce", "rms")),
            normalize=str(dataset.get("normalize", "sample_zscore")),
            seed=seed,
            expected_shape=tuple(int(v) for v in dataset.get("expected_shape", [10000, 12])),
            data_root=root,
        )
        train = BJDASMatDataset(split="train", **common)
        val = BJDASMatDataset(split="val", **common)
        test = BJDASMatDataset(split="test", **common)
        eval_batch_size = int(dataset.get("eval_sample_batch_size", batch_size))
        val.eval_window_batch_size = int(dataset.get("eval_window_batch_size", batch_size))
        test.eval_window_batch_size = int(dataset.get("eval_window_batch_size", batch_size))
        return {
            "train": _make_loader(train, batch_size, True, training, seed, 0),
            "val": _make_loader(val, eval_batch_size, False, training, seed, 1),
            "test": _make_loader(test, eval_batch_size, False, training, seed, 2),
        }
    raise ValueError(f"Unsupported dataset_name={dataset.get('name', dataset.get('dataset_name'))}.")


def read_label_names(source: str | Path | dict[str, Any], num_classes: int) -> list[str]:
    if isinstance(source, dict):
        dataset = source
        mapping_candidates: list[Path] = []
        if dataset.get("class_mapping"):
            mapping_candidates.append(_resolve_path(dataset["class_mapping"], base=RELEASE_ROOT))
        if dataset.get("split_dir"):
            split_dir = _resolve_path(dataset["split_dir"], base=RELEASE_ROOT)
            mapping_candidates.append(split_dir / "class_mapping.json")
        if dataset.get("data_root"):
            mapping_candidates.append(Path(dataset["data_root"]).resolve() / "label_mapping.json")
        for mapping_path in mapping_candidates:
            if mapping_path.exists():
                mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
                return [str(mapping[str(i)]) for i in range(num_classes)]
        if dataset.get("manifest"):
            manifest_path = _resolve_path(dataset["manifest"], base=RELEASE_ROOT)
            if manifest_path.exists():
                df = pd.read_csv(manifest_path)
                label_col = "label_id" if "label_id" in df.columns else "label"
                name_col = "label_name" if "label_name" in df.columns else "class_name" if "class_name" in df.columns else None
                if label_col in df.columns and name_col is not None:
                    mapping = (
                        df[[label_col, name_col]]
                        .drop_duplicates()
                        .sort_values(label_col)
                        .set_index(label_col)[name_col]
                        .to_dict()
                    )
                    return [str(mapping.get(i, i)) for i in range(num_classes)]
        return [str(i) for i in range(num_classes)]
    mapping_path = Path(source) / "class_mapping.json"
    if mapping_path.exists():
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        return [str(mapping[str(i)]) for i in range(num_classes)]
    return [str(i) for i in range(num_classes)]
