from __future__ import annotations

import numpy as np


def sample_zscore(window: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    x = np.asarray(window, dtype=np.float32)
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return ((x - mean) / np.maximum(std, eps)).astype(np.float32, copy=False)
