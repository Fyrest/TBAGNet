from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score


def softmax_probabilities(logits: torch.Tensor) -> torch.Tensor:
    shifted = logits - logits.amax(dim=1, keepdim=True)
    exp = torch.exp(shifted)
    return exp / exp.sum(dim=1, keepdim=True).clamp_min(torch.finfo(exp.dtype).eps)


def compute_classification_metrics(y_true: Iterable[int], y_pred: Iterable[int], num_classes: int) -> dict:
    y_true = np.asarray(list(y_true), dtype=np.int64)
    y_pred = np.asarray(list(y_pred), dtype=np.int64)
    labels = list(range(int(num_classes)))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_precision": float(precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
    }


def prediction_hash(y_true, y_pred, probs=None) -> str:
    h = hashlib.sha256()
    h.update(np.asarray(list(y_true), dtype=np.int64).tobytes())
    h.update(np.asarray(list(y_pred), dtype=np.int64).tobytes())
    if probs is not None:
        h.update(np.ascontiguousarray(np.asarray(probs, dtype=np.float32)).tobytes())
    return h.hexdigest()


def save_confusion_matrix_csv(matrix, label_names, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(matrix, index=label_names, columns=label_names)
    df.index.name = "actual\\predicted"
    df.to_csv(path, encoding="utf-8-sig")


def plot_confusion_matrix(matrix, label_names, path, title="Confusion Matrix"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(4, len(label_names) * 0.9), max(3.5, len(label_names) * 0.8)))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(label_names)))
    ax.set_yticks(range(len(label_names)))
    ax.set_xticklabels(label_names, rotation=45, ha="right")
    ax.set_yticklabels(label_names)
    vmax = matrix.max() if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(int(matrix[i, j])), ha="center", va="center", color="white" if vmax and matrix[i, j] > vmax * 0.5 else "black", fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_prediction_tables(result_dir, y_true, y_pred, probs, label_names, metadata):
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    rows = {
        "sample_id": [m.get("sample_id", f"test_{i:06d}") for i, m in enumerate(metadata)],
        "file_name": [m.get("file_name", "") for m in metadata],
        "true_label": np.asarray(y_true, dtype=np.int64),
        "pred_label": np.asarray(y_pred, dtype=np.int64),
        "sample_index": np.arange(len(y_true), dtype=np.int64),
        "label": np.asarray(y_true, dtype=np.int64),
        "prediction": np.asarray(y_pred, dtype=np.int64),
    }
    for i in range(np.asarray(probs).shape[1]):
        rows[f"prob_{i}"] = probs[:, i]
    pd.DataFrame(rows).to_csv(result_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(result_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    rows2 = {
        "sample_id": rows["sample_id"],
        "file_name": rows["file_name"],
        "sample_index": rows["sample_index"],
        "true_label": rows["true_label"],
        "predicted_label": np.asarray(y_pred, dtype=np.int64),
        "pred_label": np.asarray(y_pred, dtype=np.int64),
        "true_class": [label_names[i] for i in y_true],
        "predicted_class": [label_names[i] for i in y_pred],
    }
    for i, name in enumerate(label_names):
        rows2[f"probability_{name}"] = probs[:, i]
    pd.DataFrame(rows2).to_csv(result_dir / "sample_predictions_with_classes.csv", index=False, encoding="utf-8-sig")


def save_per_class_metrics(path, y_true, y_pred, label_names):
    report = classification_report(y_true, y_pred, labels=list(range(len(label_names))), target_names=label_names, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(path, encoding="utf-8-sig")
