"""Vector similarity search. Uses FAISS when available, otherwise an exact NumPy index."""
from __future__ import annotations

from typing import Sequence

import numpy as np

try:  # faiss-cpu has wheels for Python 3.13 from v1.10; fall back gracefully if it is missing
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None


class VectorIndex:
    """Inner-product index over L2-normalised vectors (inner product == cosine similarity)."""

    def __init__(self, dim: int, use_faiss: bool = True):
        self.dim = dim
        self.backend = "faiss" if (faiss is not None and use_faiss) else "numpy"
        if self.backend == "faiss":
            self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
        else:
            self._ids = np.empty(0, dtype=np.int64)
            self._mat = np.empty((0, dim), dtype=np.float32)

    def __len__(self) -> int:
        return int(self._index.ntotal) if self.backend == "faiss" else len(self._ids)

    def add(self, ids: Sequence[int], vectors: np.ndarray) -> None:
        if len(ids) == 0:
            return
        vecs = np.ascontiguousarray(vectors, dtype=np.float32)
        if vecs.ndim != 2 or vecs.shape[1] != self.dim:
            raise ValueError(f"Expected vectors of dimension {self.dim}, got {vecs.shape}")
        id_array = np.asarray(ids, dtype=np.int64)
        if self.backend == "faiss":
            self._index.add_with_ids(vecs, id_array)
        else:
            self._ids = np.concatenate([self._ids, id_array])
            self._mat = np.vstack([self._mat, vecs])

    def remove(self, ids: Sequence[int]) -> None:
        if len(ids) == 0:
            return
        id_array = np.asarray(ids, dtype=np.int64)
        if self.backend == "faiss":
            self._index.remove_ids(id_array)
        else:
            keep = ~np.isin(self._ids, id_array)
            self._ids, self._mat = self._ids[keep], self._mat[keep]

    def search(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        total = len(self)
        if total == 0 or k <= 0:
            return []
        k = min(k, total)
        q = np.ascontiguousarray(np.asarray(query, dtype=np.float32).reshape(1, -1))
        if self.backend == "faiss":
            scores, ids = self._index.search(q, k)
            return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]
        sims = self._mat @ q[0]
        order = np.argsort(-sims)[:k]
        return [(int(self._ids[i]), float(sims[i])) for i in order]
