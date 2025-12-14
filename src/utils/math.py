from __future__ import annotations

import numpy as np


def l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize a vector (or 2D array row-wise) safely."""
    arr = np.asarray(vec, dtype=np.float32)
    if arr.ndim == 1:
        denom = float(np.linalg.norm(arr))
        if denom < eps:
            return arr
        return arr / denom
    if arr.ndim == 2:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.maximum(norms, eps)
        return arr / norms
    raise ValueError(f"Unsupported ndim={arr.ndim}")


def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    """Cosine similarity for 1D vectors."""
    va = np.asarray(a, dtype=np.float32).reshape(-1)
    vb = np.asarray(b, dtype=np.float32).reshape(-1)
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))
