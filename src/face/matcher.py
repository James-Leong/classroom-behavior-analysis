from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from src.utils.math import l2_normalize


@dataclass
class MatcherConfig:
    # Base cosine threshold used for high-quality faces.
    threshold: float = 0.40
    # Require Top-1 to be sufficiently above Top-2 to avoid ambiguous assignments.
    margin: float = 0.03
    # Quality adaptive threshold: effective = threshold - alpha * (quality - 0.5)
    # (Higher quality => slightly lower effective threshold).
    quality_alpha: float = 0.10
    min_threshold: float = 0.30
    max_threshold: float = 0.65
    unknown_label: str = "未知"


class CosineMatcher:
    """Vectorized cosine matcher for small/medium galleries.

    Assumes embeddings are L2-normalized.
    """

    def __init__(self, config: MatcherConfig):
        self.config = config

        # Cache a flattened gallery index to avoid Python loops:
        # - matrix: (N, D) float32
        # - person_ids: (N,) int32, mapping row -> person index
        # - names: list[str] length P
        self._cache_key: Optional[Tuple[int, int, int, int]] = None
        self._cache_names: List[str] = []
        self._cache_person_ids: Optional[np.ndarray] = None
        self._cache_matrix: Optional[np.ndarray] = None

        # Optional torch/CUDA backend (auto): if CUDA is available, use GPU matmul + reduce.
        # This preserves matching semantics (per-person best among K embeddings) while avoiding CPU bottlenecks.
        self._cache_matrix_t = None
        self._cache_person_ids_t = None
        self._cache_device: Optional[str] = None

    def _build_index(self, person_to_embeddings: Dict[str, np.ndarray]) -> None:
        names: List[str] = []
        mats: List[np.ndarray] = []
        ids: List[np.ndarray] = []

        total_rows = 0
        dim = 0

        for name, embs in person_to_embeddings.items():
            if embs is None:
                continue
            mat = np.asarray(embs, dtype=np.float32)
            if mat.ndim == 1:
                mat = mat.reshape(1, -1)
            if mat.size == 0:
                continue

            if dim == 0:
                dim = int(mat.shape[1])
            # Skip incompatible dims defensively
            if int(mat.shape[1]) != dim:
                continue

            names.append(str(name))
            mats.append(mat)
            ids.append(np.full((int(mat.shape[0]),), int(len(names) - 1), dtype=np.int32))
            total_rows += int(mat.shape[0])

        if not mats:
            self._cache_key = None
            self._cache_names = []
            self._cache_person_ids = None
            self._cache_matrix = None
            return

        self._cache_names = names
        self._cache_matrix = np.ascontiguousarray(np.concatenate(mats, axis=0).astype(np.float32, copy=False))
        self._cache_person_ids = np.concatenate(ids, axis=0).astype(np.int32, copy=False)

        # Torch cache is built lazily (only when CUDA is available).
        self._cache_matrix_t = None
        self._cache_person_ids_t = None
        self._cache_device = None

    def _ensure_index(self, person_to_embeddings: Dict[str, np.ndarray]) -> None:
        if not person_to_embeddings:
            self._cache_key = None
            self._cache_names = []
            self._cache_person_ids = None
            self._cache_matrix = None
            return

        # Heuristic key: object identity + person count + total rows + embedding dim.
        # This avoids rebuilding on hot path while remaining robust for common updates.
        try:
            persons = int(len(person_to_embeddings))
        except Exception:
            persons = 0
        total_rows = 0
        dim = 0
        try:
            for embs in person_to_embeddings.values():
                if embs is None:
                    continue
                mat = np.asarray(embs)
                if mat.ndim == 1:
                    mat = mat.reshape(1, -1)
                if mat.size == 0:
                    continue
                total_rows += int(mat.shape[0])
                if dim == 0:
                    dim = int(mat.shape[1])
        except Exception:
            pass

        key = (int(id(person_to_embeddings)), persons, int(total_rows), int(dim))
        if self._cache_key == key and self._cache_matrix is not None and self._cache_person_ids is not None:
            return
        self._cache_key = key
        self._build_index(person_to_embeddings)

    def _auto_device(self) -> str:
        if torch is None:
            return "cpu"
        try:
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _ensure_torch_index(self) -> bool:
        if torch is None:
            return False
        device = self._auto_device()
        if device != "cuda":
            return False
        if self._cache_matrix is None or self._cache_person_ids is None:
            return False
        if self._cache_matrix_t is not None and self._cache_person_ids_t is not None and self._cache_device == device:
            return True
        try:
            mat = torch.from_numpy(self._cache_matrix)
            pids = torch.from_numpy(self._cache_person_ids.astype(np.int64, copy=False))
            mat = mat.to(device, non_blocking=True)
            pids = pids.to(device, non_blocking=True)
            self._cache_matrix_t = mat
            self._cache_person_ids_t = pids
            self._cache_device = device
            return True
        except Exception:
            self._cache_matrix_t = None
            self._cache_person_ids_t = None
            self._cache_device = None
            return False

    def effective_threshold(self, quality: float) -> float:
        q = float(quality)
        thr = float(self.config.threshold) - float(self.config.quality_alpha) * (q - 0.5)
        thr = max(self.config.min_threshold, min(self.config.max_threshold, thr))
        return thr

    def match(
        self,
        embedding: np.ndarray,
        person_to_embeddings: Dict[str, np.ndarray],
        quality: float,
        topk_debug: int = 5,
    ) -> Tuple[str, float, List[Tuple[str, float]]]:
        """Return (identity, best_similarity, topk_per_person).

        `person_to_embeddings` maps person -> (K, D) matrix of normalized embeddings.
        """
        if not person_to_embeddings:
            return self.config.unknown_label, 0.0, []

        q = l2_normalize(np.asarray(embedding, dtype=np.float32).reshape(-1))

        self._ensure_index(person_to_embeddings)
        if self._cache_matrix is None or self._cache_person_ids is None or not self._cache_names:
            return self.config.unknown_label, 0.0, []

        # Prefer auto torch/CUDA backend when available
        if self._ensure_torch_index():
            try:
                device = self._auto_device()
                q_t = torch.from_numpy(q.astype(np.float32, copy=False))
                q_t = q_t.to(device, non_blocking=True)
                sims_t = self._cache_matrix_t @ q_t

                best_t = torch.full((len(self._cache_names),), -1e9, dtype=sims_t.dtype, device=sims_t.device)
                # best[person] = max(best[person], sims[row])
                if hasattr(best_t, "scatter_reduce_"):
                    best_t.scatter_reduce_(0, self._cache_person_ids_t, sims_t, reduce="amax", include_self=True)
                else:
                    # Fallback: index_reduce_ (older torch)
                    if hasattr(best_t, "index_reduce_"):
                        best_t.index_reduce_(0, self._cache_person_ids_t, sims_t, reduce="amax")
                    else:
                        raise RuntimeError("torch scatter_reduce not available")

                order_t = torch.argsort(best_t, descending=True)
                best_idx = int(order_t[0].item())
                best_name = self._cache_names[best_idx]
                best_sim = float(best_t[best_idx].item())
                second_sim = float(best_t[int(order_t[1].item())].item()) if int(order_t.numel()) >= 2 else -1.0

                eff_thr = self.effective_threshold(quality)
                topk = int(max(1, topk_debug))
                topk_idx = order_t[:topk].detach().cpu().numpy().astype(int)
                best_cpu = best_t.detach().cpu().numpy().astype(np.float32, copy=False)
                topk_list = [(self._cache_names[int(i)], float(best_cpu[int(i)])) for i in topk_idx]

                if best_sim >= eff_thr and (best_sim - second_sim) >= float(self.config.margin):
                    return best_name, best_sim, topk_list
                return self.config.unknown_label, best_sim, topk_list
            except Exception:
                # Fall back to numpy path
                pass

        # Vectorized similarity: one matmul + one segmented max (CPU)
        sims = self._cache_matrix @ q
        best_per_person = np.full((len(self._cache_names),), -1e9, dtype=np.float32)
        np.maximum.at(best_per_person, self._cache_person_ids, sims.astype(np.float32, copy=False))

        order = np.argsort(-best_per_person)
        best_idx = int(order[0])
        best_name = self._cache_names[best_idx]
        best_sim = float(best_per_person[best_idx])

        second_sim = float(best_per_person[int(order[1])]) if len(order) >= 2 else -1.0
        eff_thr = self.effective_threshold(quality)

        topk = int(max(1, topk_debug))
        topk_list = [(self._cache_names[int(i)], float(best_per_person[int(i)])) for i in order[:topk]]

        if best_sim >= eff_thr and (best_sim - second_sim) >= float(self.config.margin):
            return best_name, best_sim, topk_list

        return self.config.unknown_label, best_sim, topk_list
