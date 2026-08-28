from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.io import loadmat


SPLIT_NAMES = ("train", "val", "test")
DEFAULT_RATIOS = (0.71, 0.083, 0.207)


@dataclass(frozen=True)
class SourceRecord:
    relative_path: str
    class_id: int
    class_name: str
    length: int
    source_signal_sha256: str


def load_railway_signal(path: str | Path, data_key: str = "signal") -> np.ndarray:
    """Load a Railway-DAS signal using the release Dataset precision rules."""
    path = Path(path)
    payload = loadmat(path)
    if data_key in payload:
        signal = payload[data_key]
    else:
        numeric = [
            value
            for key, value in payload.items()
            if not key.startswith("__") and np.issubdtype(np.asarray(value).dtype, np.number)
        ]
        if not numeric:
            raise ValueError(f"No numeric signal found in {path}.")
        signal = numeric[0]
    signal = np.asarray(signal)
    if np.iscomplexobj(signal):
        signal = np.real(signal)
    signal = np.squeeze(signal).astype(np.float32, copy=False)
    if signal.ndim != 1:
        raise ValueError(f"Expected a 1D Railway-DAS signal in {path}, got {signal.shape}.")
    if not np.isfinite(signal).all():
        raise ValueError(f"Railway-DAS signal contains NaN or Inf: {path}")
    return np.ascontiguousarray(signal, dtype=np.float32)


def stable_array_sha256(array: np.ndarray) -> str:
    """Hash canonical shape, dtype, and contiguous content."""
    value = np.ascontiguousarray(np.asarray(array, dtype=np.float32))
    header = json.dumps(
        {"dtype": value.dtype.str, "shape": list(value.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\n")
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def candidate_starts(length: int, window_length: int = 1024, stride: int = 512) -> list[int]:
    """Return the nominal Railway-DAS candidate grid without tail interpolation."""
    if window_length <= 0 or stride <= 0:
        raise ValueError("window_length and stride must be positive.")
    if length < window_length:
        return []
    return list(range(0, length - window_length + 1, stride))


def _read_class_mapping(path: str | Path) -> dict[int, str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    mapping = {int(key): str(value) for key, value in payload.items()}
    if sorted(mapping) != list(range(len(mapping))):
        raise ValueError("Class mapping keys must be contiguous integer IDs starting at zero.")
    return mapping


def _is_public_relative_path(value: str) -> bool:
    path = Path(value)
    windows_path = PureWindowsPath(value)
    return not path.is_absolute() and not windows_path.is_absolute() and ".." not in path.parts


def _catalog_windows(
    catalog_path: str | Path,
    window_length: int,
    stride: int,
) -> dict[str, list[tuple[int, int]]]:
    frame = pd.read_csv(catalog_path)
    required = {"relative_path", "start", "end"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Window catalog is missing columns: {sorted(missing)}")
    windows: dict[str, list[tuple[int, int]]] = defaultdict(list)
    seen: set[tuple[str, int, int]] = set()
    for row in frame.itertuples(index=False):
        relative_path = str(row.relative_path).replace("\\", "/")
        start = int(row.start)
        end = int(row.end)
        key = (relative_path, start, end)
        if not _is_public_relative_path(relative_path):
            raise ValueError(f"Window catalog contains a non-relative path: {relative_path}")
        if end - start != window_length:
            raise ValueError(f"Catalog window {key} does not have length {window_length}.")
        if start < 0 or start % stride != 0:
            raise ValueError(f"Catalog window {key} is not on the nominal stride-{stride} grid.")
        if key in seen:
            raise ValueError(f"Duplicate catalog window: {key}")
        seen.add(key)
        windows[relative_path].append((start, end))
    return {key: sorted(value) for key, value in windows.items()}


def discover_sources(
    data_root: str | Path,
    class_mapping_path: str | Path,
    data_key: str = "signal",
    selected_relative_paths: Iterable[str] | None = None,
) -> tuple[list[SourceRecord], dict[str, np.ndarray]]:
    data_root = Path(data_root).resolve()
    mapping = _read_class_mapping(class_mapping_path)
    selected = None if selected_relative_paths is None else {str(value).replace("\\", "/") for value in selected_relative_paths}
    sources: list[SourceRecord] = []
    signals: dict[str, np.ndarray] = {}
    for class_id, class_name in mapping.items():
        class_dir = data_root / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Railway-DAS class directory not found: {class_dir}")
        for path in sorted(class_dir.glob("*.mat"), key=lambda item: item.name.casefold()):
            relative_path = path.relative_to(data_root).as_posix()
            if selected is not None and relative_path not in selected:
                continue
            signal = load_railway_signal(path, data_key=data_key)
            signals[relative_path] = signal
            sources.append(
                SourceRecord(
                    relative_path=relative_path,
                    class_id=class_id,
                    class_name=class_name,
                    length=int(signal.shape[0]),
                    source_signal_sha256=stable_array_sha256(signal),
                )
            )
    if selected is not None:
        missing = selected - set(signals)
        if missing:
            raise FileNotFoundError(f"Window catalog references missing MAT files: {sorted(missing)[:10]}")
    if not sources:
        raise ValueError("No Railway-DAS MAT sources were discovered.")
    return sources, signals


def group_sources(sources: Iterable[SourceRecord]) -> list[dict[str, Any]]:
    grouped: dict[str, list[SourceRecord]] = defaultdict(list)
    for source in sources:
        grouped[source.source_signal_sha256].append(source)
    groups: list[dict[str, Any]] = []
    for digest, members in sorted(grouped.items()):
        class_ids = {member.class_id for member in members}
        class_names = {member.class_name for member in members}
        if len(class_ids) != 1 or len(class_names) != 1:
            raise ValueError(
                f"Identical source signal {digest} occurs under conflicting classes: "
                f"{sorted((member.relative_path, member.class_name) for member in members)}"
            )
        groups.append(
            {
                "source_group_id": f"source-{digest[:16]}",
                "source_signal_sha256": digest,
                "class_id": next(iter(class_ids)),
                "class_name": next(iter(class_names)),
                "members": sorted(members, key=lambda item: item.relative_path),
            }
        )
    return groups


def build_isolation_components(
    groups: list[dict[str, Any]],
    windows_for_source: dict[str, list[tuple[int, int]]],
    signals: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Link source groups that contain an identical selected window."""
    parent = {str(group["source_signal_sha256"]): str(group["source_signal_sha256"]) for group in groups}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        keep, merge = sorted((left_root, right_root))
        parent[merge] = keep

    digest_for_source = {
        source.relative_path: str(group["source_signal_sha256"])
        for group in groups
        for source in group["members"]
    }
    owner_for_window_hash: dict[str, str] = {}
    for relative_path in sorted(windows_for_source):
        source_digest = digest_for_source[relative_path]
        signal = signals[relative_path]
        for start, end in windows_for_source[relative_path]:
            window_hash = stable_array_sha256(signal[start:end])
            previous = owner_for_window_hash.get(window_hash)
            if previous is None:
                owner_for_window_hash[window_hash] = source_digest
            else:
                union(source_digest, previous)

    component_members: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        component_members[find(str(group["source_signal_sha256"]))].append(group)
    components: list[dict[str, Any]] = []
    source_to_component: dict[str, str] = {}
    for _, member_groups in sorted(component_members.items()):
        source_digests = sorted(str(group["source_signal_sha256"]) for group in member_groups)
        class_ids = {int(group["class_id"]) for group in member_groups}
        class_names = {str(group["class_name"]) for group in member_groups}
        if len(class_ids) != 1 or len(class_names) != 1:
            raise ValueError(
                "An exact window occurs under conflicting Railway-DAS classes: "
                f"{[(digest, group['class_name']) for digest, group in zip(source_digests, member_groups)]}"
            )
        component_digest = hashlib.sha256("|".join(source_digests).encode("ascii")).hexdigest()
        components.append(
            {
                "source_signal_sha256": component_digest,
                "class_id": next(iter(class_ids)),
                "class_name": next(iter(class_names)),
                "source_digests": source_digests,
            }
        )
        for source_digest in source_digests:
            source_to_component[source_digest] = component_digest
    return components, source_to_component


def _largest_remainder_counts(total: int, ratios: tuple[float, float, float]) -> dict[str, int]:
    raw = np.asarray(ratios, dtype=np.float64) * int(total)
    counts = np.floor(raw).astype(int)
    for index in np.argsort(-(raw - counts))[: int(total - counts.sum())]:
        counts[index] += 1
    if total >= len(SPLIT_NAMES):
        for index in range(len(SPLIT_NAMES)):
            if counts[index] == 0:
                donor = int(np.argmax(counts))
                counts[donor] -= 1
                counts[index] += 1
    return {name: int(counts[index]) for index, name in enumerate(SPLIT_NAMES)}


def assign_source_groups(
    groups: list[dict[str, Any]],
    window_counts: dict[str, int],
    seed: int,
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
) -> dict[str, str]:
    ratios_array = np.asarray(ratios, dtype=np.float64)
    if ratios_array.shape != (3,) or np.any(ratios_array <= 0):
        raise ValueError("ratios must contain three positive values.")
    ratios_array /= ratios_array.sum()
    normalized_ratios = tuple(float(value) for value in ratios_array)
    assignments: dict[str, str] = {}
    by_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        by_class[int(group["class_id"])].append(group)
    for class_id, class_groups in sorted(by_class.items()):
        target_group_counts = _largest_remainder_counts(len(class_groups), normalized_ratios)
        total_windows = sum(int(window_counts[group["source_signal_sha256"]]) for group in class_groups)
        target_windows = {
            name: total_windows * normalized_ratios[index] for index, name in enumerate(SPLIT_NAMES)
        }
        assigned_groups = {name: 0 for name in SPLIT_NAMES}
        assigned_windows = {name: 0 for name in SPLIT_NAMES}

        def deterministic_key(group: dict[str, Any]) -> str:
            text = f"{seed}|{class_id}|{group['source_signal_sha256']}"
            return hashlib.sha256(text.encode("utf-8")).hexdigest()

        ordered = sorted(
            class_groups,
            key=lambda group: (-int(window_counts[group["source_signal_sha256"]]), deterministic_key(group)),
        )

        def objective(
            group_counts: dict[str, int],
            split_windows: dict[str, int],
        ) -> float:
            window_scale = max(float(total_windows * total_windows), 1.0)
            group_scale = max(float(len(class_groups) * len(class_groups)), 1.0)
            window_error = sum(
                (float(split_windows[name]) - float(target_windows[name])) ** 2
                for name in SPLIT_NAMES
            ) / window_scale
            group_error = sum(
                (float(group_counts[name]) - float(target_group_counts[name])) ** 2
                for name in SPLIT_NAMES
            ) / group_scale
            return window_error + 0.02 * group_error

        group_by_digest = {str(group["source_signal_sha256"]): group for group in class_groups}
        class_assignments: dict[str, str] = {}
        for position, group in enumerate(ordered):
            digest = str(group["source_signal_sha256"])
            weight = int(window_counts[digest])
            remaining_after = len(ordered) - position - 1
            empty_splits = [name for name in SPLIT_NAMES if assigned_groups[name] == 0]
            choices = list(SPLIT_NAMES)
            if remaining_after < len(empty_splits):
                choices = empty_splits

            def score(name: str) -> tuple[float, int]:
                projected_groups = dict(assigned_groups)
                projected_windows = dict(assigned_windows)
                projected_groups[name] += 1
                projected_windows[name] += weight
                return objective(projected_groups, projected_windows), SPLIT_NAMES.index(name)

            selected_split = min(choices, key=score)
            class_assignments[digest] = selected_split
            assigned_groups[selected_split] += 1
            assigned_windows[selected_split] += weight

        # Deterministic single-group local search improves window-count fit while
        # retaining a small source-group balance penalty.
        for _ in range(20):
            improved = False
            current_score = objective(assigned_groups, assigned_windows)
            for digest in sorted(class_assignments, key=lambda value: deterministic_key(group_by_digest[value])):
                current_split = class_assignments[digest]
                if assigned_groups[current_split] <= 1:
                    continue
                weight = int(window_counts[digest])
                best_split = current_split
                best_score = current_score
                for candidate_split in SPLIT_NAMES:
                    if candidate_split == current_split:
                        continue
                    projected_groups = dict(assigned_groups)
                    projected_windows = dict(assigned_windows)
                    projected_groups[current_split] -= 1
                    projected_groups[candidate_split] += 1
                    projected_windows[current_split] -= weight
                    projected_windows[candidate_split] += weight
                    candidate_score = objective(projected_groups, projected_windows)
                    if candidate_score + 1e-15 < best_score:
                        best_score = candidate_score
                        best_split = candidate_split
                if best_split != current_split:
                    class_assignments[digest] = best_split
                    assigned_groups[current_split] -= 1
                    assigned_groups[best_split] += 1
                    assigned_windows[current_split] -= weight
                    assigned_windows[best_split] += weight
                    current_score = best_score
                    improved = True
            if not improved:
                break
        assignments.update(class_assignments)
    return assignments


def _pairwise_intersections(values: dict[str, set[str]]) -> dict[str, int]:
    return {
        "train_val": len(values["train"] & values["val"]),
        "train_test": len(values["train"] & values["test"]),
        "val_test": len(values["val"] & values["test"]),
    }


def validate_grouped_manifest(
    manifest: pd.DataFrame,
    data_root: str | Path,
    data_key: str = "signal",
) -> dict[str, Any]:
    required = {
        "relative_path",
        "label",
        "label_name",
        "start",
        "end",
        "source_signal_sha256",
        "source_group_id",
        "window_sha256",
        "split",
        "seed",
    }
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Grouped manifest is missing columns: {sorted(missing)}")
    invalid_splits = set(manifest["split"].astype(str)) - set(SPLIT_NAMES)
    if invalid_splits:
        raise ValueError(f"Unexpected split names: {sorted(invalid_splits)}")
    for relative_path in manifest["relative_path"].astype(str).unique():
        if not _is_public_relative_path(relative_path):
            raise ValueError(f"Manifest contains a non-relative path: {relative_path}")

    source_sets = {
        split: set(manifest.loc[manifest["split"] == split, "source_signal_sha256"].astype(str))
        for split in SPLIT_NAMES
    }
    source_intersections = _pairwise_intersections(source_sets)
    if any(source_intersections.values()):
        raise RuntimeError(f"Source-signal hash isolation failed: {source_intersections}")
    source_split_counts = manifest.groupby("source_signal_sha256")["split"].nunique()
    if int(source_split_counts.max()) != 1:
        raise RuntimeError("A source_signal_sha256 is assigned to multiple splits.")

    data_root = Path(data_root).resolve()
    signal_cache: dict[str, np.ndarray] = {}
    computed_window_sets = {split: set() for split in SPLIT_NAMES}
    for row in manifest.itertuples(index=False):
        relative_path = str(row.relative_path)
        if relative_path not in signal_cache:
            signal_cache[relative_path] = load_railway_signal(data_root / relative_path, data_key=data_key)
        signal = signal_cache[relative_path]
        computed_source_hash = stable_array_sha256(signal)
        if computed_source_hash != str(row.source_signal_sha256):
            raise RuntimeError(f"Stored source hash does not match {relative_path}.")
        start = int(row.start)
        end = int(row.end)
        if start < 0 or end <= start or end > signal.shape[0]:
            raise ValueError(f"Invalid source interval for {relative_path}: [{start}, {end})")
        computed_window_hash = stable_array_sha256(signal[start:end])
        if computed_window_hash != str(row.window_sha256):
            raise RuntimeError(f"Stored window hash does not match {relative_path}[{start}:{end}].")
        computed_window_sets[str(row.split)].add(computed_window_hash)
    window_intersections = _pairwise_intersections(computed_window_sets)
    if any(window_intersections.values()):
        raise RuntimeError(f"Exact-window hash isolation failed: {window_intersections}")
    return {
        "source_hash_intersections": source_intersections,
        "window_hash_intersections": window_intersections,
        "rows": int(len(manifest)),
        "unique_source_groups": int(manifest["source_signal_sha256"].nunique()),
    }


def generate_grouped_split(
    data_root: str | Path,
    class_mapping_path: str | Path,
    output_dir: str | Path,
    *,
    window_catalog: str | Path | None = None,
    seed: int = 42,
    window_length: int = 1024,
    stride: int = 512,
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    data_key: str = "signal",
    protected_output_dir: str | Path | None = None,
) -> dict[str, Any]:
    data_root = Path(data_root).resolve()
    output_dir = Path(output_dir).resolve()
    if protected_output_dir is not None and output_dir == Path(protected_output_dir).resolve():
        raise ValueError("Refusing to overwrite the release fixed Railway-DAS split directory.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be absent or empty: {output_dir}")

    catalog = None
    selected_paths = None
    if window_catalog is not None:
        catalog = _catalog_windows(window_catalog, window_length=window_length, stride=stride)
        selected_paths = set(catalog)
    sources, signals = discover_sources(
        data_root,
        class_mapping_path,
        data_key=data_key,
        selected_relative_paths=selected_paths,
    )
    groups = group_sources(sources)

    windows_for_source: dict[str, list[tuple[int, int]]] = {}
    for source in sources:
        if catalog is not None:
            windows = list(catalog[source.relative_path])
        else:
            windows = [
                (start, start + window_length)
                for start in candidate_starts(source.length, window_length=window_length, stride=stride)
            ]
        for start, end in windows:
            if end > source.length:
                raise ValueError(f"Window exceeds source length: {source.relative_path}[{start}:{end}]")
        windows_for_source[source.relative_path] = windows

    source_window_counts = {
        str(group["source_signal_sha256"]): sum(
            len(windows_for_source[source.relative_path]) for source in group["members"]
        )
        for group in groups
    }
    components, source_to_component = build_isolation_components(
        groups,
        windows_for_source=windows_for_source,
        signals=signals,
    )
    component_window_counts = {
        str(component["source_signal_sha256"]): sum(
            source_window_counts[source_digest] for source_digest in component["source_digests"]
        )
        for component in components
    }
    component_assignments = assign_source_groups(
        components,
        window_counts=component_window_counts,
        seed=seed,
        ratios=ratios,
    )
    assignments = {
        source_digest: component_assignments[component_digest]
        for source_digest, component_digest in source_to_component.items()
    }

    rows: list[dict[str, Any]] = []
    for group in groups:
        digest = str(group["source_signal_sha256"])
        split = assignments[digest]
        for source in group["members"]:
            signal = signals[source.relative_path]
            for start, end in windows_for_source[source.relative_path]:
                rows.append(
                    {
                        "relative_path": source.relative_path,
                        "file_name": Path(source.relative_path).name,
                        "class_id": source.class_id,
                        "class_name": source.class_name,
                        "label": source.class_id,
                        "label_name": source.class_name,
                        "start": int(start),
                        "end": int(end),
                        "source_group_id": str(group["source_group_id"]),
                        "source_signal_sha256": digest,
                        "window_sha256": stable_array_sha256(signal[start:end]),
                        "split": split,
                        "seed": int(seed),
                    }
                )
    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise ValueError("Grouped split generation produced no windows.")
    manifest = manifest.sort_values(
        ["split", "class_id", "source_group_id", "relative_path", "start"],
        kind="stable",
    ).reset_index(drop=True)
    validation = validate_grouped_manifest(manifest, data_root=data_root, data_key=data_key)

    output_dir.mkdir(parents=True, exist_ok=True)
    split_frames: dict[str, pd.DataFrame] = {}
    for split in SPLIT_NAMES:
        frame = manifest[manifest["split"] == split].reset_index(drop=True).copy()
        frame.insert(0, "sample_id", [f"{split}_{index:06d}" for index in range(len(frame))])
        frame.to_csv(output_dir / f"{split}.csv", index=False, encoding="utf-8-sig")
        split_frames[split] = frame
    manifest.to_csv(output_dir / "manifest.csv", index=False, encoding="utf-8-sig")

    class_counts = {
        split: {
            str(int(label)): int(count)
            for label, count in split_frames[split]["label"].value_counts().sort_index().items()
        }
        for split in SPLIT_NAMES
    }
    group_counts = {
        split: int(
            sum(1 for digest, assigned_split in assignments.items() if assigned_split == split)
        )
        for split in SPLIT_NAMES
    }
    metadata = {
        "dataset": "Railway-DAS",
        "generator": "content-hash-grouped-source-split",
        "seed": int(seed),
        "ratios": {name: float(ratios[index]) for index, name in enumerate(SPLIT_NAMES)},
        "window_length": int(window_length),
        "nominal_stride": int(stride),
        "nominal_overlap": int(window_length - stride),
        "window_selection": "catalog" if catalog is not None else "exhaustive_nominal_grid",
        "source_groups": group_counts,
        "isolation_components": {
            split: int(
                sum(1 for digest, assigned_split in component_assignments.items() if assigned_split == split)
            )
            for split in SPLIT_NAMES
        },
        "window_counts": {split: int(len(split_frames[split])) for split in SPLIT_NAMES},
        "class_window_counts": class_counts,
        "integrity": validation,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return metadata
