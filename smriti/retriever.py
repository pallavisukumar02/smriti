"""Hybrid retrieval: dense (embeddings + FAISS) and sparse (BM25), merged with Reciprocal Rank Fusion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bm25 import BM25
from .chunking import Chunk
from .fusion import reciprocal_rank_fusion
from .vector_store import VectorStore

MODES = ("hybrid", "semantic", "keyword")


@dataclass
class Hit:
    chunk: Chunk
    score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None

    @property
    def matched_by(self) -> str:
        parts = []
        if self.dense_rank is not None:
            parts.append(f"semantic #{self.dense_rank}")
        if self.sparse_rank is not None:
            parts.append(f"keyword #{self.sparse_rank}")
        return " · ".join(parts) or "coverage"


class HybridRetriever:
    def __init__(self, embedder):
        self.embedder = embedder
        self.chunks: list[Chunk] = []
        self.store = VectorStore(embedder.dim)
        self.bm25 = BM25()

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        self.store.add(self.embedder.encode([c.text for c in chunks]))
        self.chunks.extend(chunks)
        self.bm25.fit([f"{c.doc_name} {c.text}" for c in self.chunks])

    def remove_doc(self, doc_id: str) -> None:
        mask = np.array([c.doc_id != doc_id for c in self.chunks], dtype=bool)
        self.store.keep(mask)
        self.chunks = [c for c, m in zip(self.chunks, mask) if m]
        self.bm25.fit([f"{c.doc_name} {c.text}" for c in self.chunks])

    # ------------------------------------------------------------------ search
    def _dense(self, query: str, previous: str, n: int) -> list[int]:
        q = self.embedder.encode([query])[0]
        if previous:  # follow-ups: lean slightly towards the previous question's topic
            q = q + 0.3 * self.embedder.encode([previous])[0]
        return [i for i, _ in self.store.search(q, n)]

    def _sparse(self, query: str, previous: str, n: int) -> list[int]:
        scores = np.array(self.bm25.scores(query))
        if previous:
            scores = scores + 0.35 * np.array(self.bm25.scores(previous))
        order = np.argsort(-scores)[:n]
        return [int(i) for i in order if scores[i] > 0]

    def search(
        self,
        query: str,
        k: int = 6,
        mode: str = "hybrid",
        previous: str = "",
        candidates: int = 25,
        max_per_doc: int = 3,
    ) -> list[Hit]:
        if not self.chunks:
            return []
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")

        dense = self._dense(query, previous, candidates) if mode in ("hybrid", "semantic") else []
        sparse = self._sparse(query, previous, candidates) if mode in ("hybrid", "keyword") else []
        dense_rank = {d: r for r, d in enumerate(dense, 1)}
        sparse_rank = {d: r for r, d in enumerate(sparse, 1)}

        fused = reciprocal_rank_fusion([r for r in (dense, sparse) if r])

        hits: list[Hit] = []
        per_doc: dict[str, int] = {}
        for idx, score in fused:
            c = self.chunks[idx]
            if per_doc.get(c.doc_id, 0) >= max_per_doc and len(fused) > k:
                continue
            per_doc[c.doc_id] = per_doc.get(c.doc_id, 0) + 1
            hits.append(Hit(c, score, dense_rank.get(idx), sparse_rank.get(idx)))
            if len(hits) >= k:
                break
        return hits

    def spread(self, k: int = 8) -> list[Hit]:
        """Evenly sample chunks across documents, for broad requests like 'summarize'."""
        by_doc: dict[str, list[Chunk]] = {}
        for c in self.chunks:
            by_doc.setdefault(c.doc_id, []).append(c)
        per = max(1, k // max(1, len(by_doc)))
        out: list[Hit] = []
        for chunks in by_doc.values():
            step = max(1, len(chunks) // per)
            out.extend(Hit(c, 0.0) for c in chunks[::step][:per])
        return out[:k]
