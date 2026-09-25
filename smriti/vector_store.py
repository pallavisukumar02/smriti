"""A small vector index backed by FAISS, with a NumPy fallback if FAISS isn't installed."""

from __future__ import annotations

import numpy as np

try:
    import faiss  # type: ignore

    HAS_FAISS = True
except ImportError:  # pragma: no cover - depends on environment
    faiss = None
    HAS_FAISS = False


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype="float32")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


class VectorStore:
    """Exact inner-product search (IndexFlatIP). With normalised vectors this is cosine similarity.

    Exact search is the right call at this scale (thousands of chunks); swap in IndexHNSWFlat
    or IndexIVFFlat for millions.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.vectors = np.zeros((0, dim), dtype="float32")
        self._index = faiss.IndexFlatIP(dim) if HAS_FAISS else None

    def __len__(self) -> int:
        return self.vectors.shape[0]

    def add(self, vectors: np.ndarray) -> None:
        vectors = _normalize(vectors)
        if vectors.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {vectors.shape[1]}")
        self.vectors = np.vstack([self.vectors, vectors])
        if self._index is not None:
            self._index.add(vectors)

    def keep(self, mask: np.ndarray) -> None:
        """Keep only rows where mask is True (used to delete a document), then rebuild the index."""
        self.vectors = self.vectors[np.asarray(mask, dtype=bool)]
        if self._index is not None:
            self._index = faiss.IndexFlatIP(self.dim)
            if len(self.vectors):
                self._index.add(self.vectors)

    def search(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        if not len(self):
            return []
        q = _normalize(np.atleast_2d(query))
        k = min(k, len(self))
        if self._index is not None:
            scores, ids = self._index.search(q, k)
            return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]
        sims = self.vectors @ q[0]
        top = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in top]
