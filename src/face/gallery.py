from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from src.utils.math import l2_normalize


@dataclass
class GalleryConfig:
    # Keep at most K embeddings per person (highest-quality first).
    max_embeddings_per_person: int = 5
    # File name for persisted gallery.
    filename: str = "gallery_embeddings.pkl"
    # Schema version to support future migrations.
    schema_version: str = "v2"


class Gallery:
    """In-memory gallery with persistence.

    Stores per-person multiple embeddings (normalized). Also exposes a centroid map
    for backward compatibility.
    """

    def __init__(self, config: GalleryConfig):
        self.config = config
        self.person_to_embeddings: Dict[str, np.ndarray] = {}
        self.person_to_qualities: Dict[str, np.ndarray] = {}
        self.stats: Dict[str, dict] = {}

    @property
    def centroids(self) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for name, embs in self.person_to_embeddings.items():
            if embs is None:
                continue
            mat = np.asarray(embs, dtype=np.float32)
            if mat.ndim == 1:
                mat = mat.reshape(1, -1)
            if mat.size == 0:
                continue
            c = np.mean(mat, axis=0)
            out[name] = l2_normalize(c)
        return out

    def save(self, gallery_dir: Path, threshold: float, quality_threshold: float) -> Path:
        import pickle

        gallery_dir = Path(gallery_dir)
        gallery_dir.mkdir(parents=True, exist_ok=True)
        fp = gallery_dir / self.config.filename
        data = {
            "schema_version": self.config.schema_version,
            "threshold": float(threshold),
            "quality_threshold": float(quality_threshold),
            "person_to_embeddings": self.person_to_embeddings,
            "person_to_qualities": self.person_to_qualities,
            "stats": self.stats,
        }
        with open(fp, "wb") as f:
            pickle.dump(data, f)
        return fp

    def load(self, gallery_dir: Path) -> bool:
        import pickle

        gallery_dir = Path(gallery_dir)
        fp = gallery_dir / self.config.filename
        if not fp.exists():
            return False
        with open(fp, "rb") as f:
            data = pickle.load(f)

        # v2 format
        if isinstance(data, dict) and data.get("schema_version") == self.config.schema_version:
            self.person_to_embeddings = data.get("person_to_embeddings", {}) or {}
            self.person_to_qualities = data.get("person_to_qualities", {}) or {}
            self.stats = data.get("stats", {}) or {}
            return True

        # Backward compatibility: older code stored {embeddings: {name: emb}, stats: ..., ...}
        if isinstance(data, dict) and "embeddings" in data:
            embs = data.get("embeddings") or {}
            migrated: Dict[str, np.ndarray] = {}
            for name, emb in embs.items():
                vec = l2_normalize(np.asarray(emb, dtype=np.float32).reshape(-1))
                migrated[name] = vec.reshape(1, -1)
            self.person_to_embeddings = migrated
            self.person_to_qualities = {}
            self.stats = data.get("stats", {}) or {}
            return True

        # Unknown
        return False

    def add_person_embeddings(
        self,
        person_name: str,
        embeddings: np.ndarray,
        qualities: Optional[np.ndarray] = None,
    ) -> None:
        name = str(person_name)
        mat = np.asarray(embeddings, dtype=np.float32)
        if mat.ndim == 1:
            mat = mat.reshape(1, -1)
        if mat.size == 0:
            return

        mat = l2_normalize(mat)

        if qualities is None:
            q = np.ones((mat.shape[0],), dtype=np.float32)
        else:
            q = np.asarray(qualities, dtype=np.float32).reshape(-1)
            if q.shape[0] != mat.shape[0]:
                q = np.ones((mat.shape[0],), dtype=np.float32)

        # Keep top-K by quality
        order = np.argsort(-q)
        order = order[: int(self.config.max_embeddings_per_person)]
        mat = mat[order]
        q = q[order]

        self.person_to_embeddings[name] = mat
        self.person_to_qualities[name] = q

        self.stats[name] = {
            "count": int(mat.shape[0]),
            "quality": float(np.mean(q)) if q.size else 0.0,
        }
