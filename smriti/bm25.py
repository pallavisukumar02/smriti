"""Okapi BM25 implemented from scratch (sparse, keyword retrieval)."""

from __future__ import annotations

import math
from collections import Counter

from .text import tokenize


class BM25:
    def __init__(self, k1: float = 1.4, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs: list[Counter] = []
        self.lengths: list[int] = []
        self.df: Counter = Counter()
        self.avgdl = 0.0

    def fit(self, texts: list[str]) -> "BM25":
        self.docs, self.lengths, self.df = [], [], Counter()
        for t in texts:
            toks = tokenize(t)
            tf = Counter(toks)
            self.docs.append(tf)
            self.lengths.append(len(toks))
            self.df.update(tf.keys())
        self.avgdl = sum(self.lengths) / max(1, len(self.lengths))
        return self

    def idf(self, term: str) -> float:
        n, N = self.df.get(term, 0), len(self.docs)
        return math.log(1 + (N - n + 0.5) / (n + 0.5))

    def scores(self, query: str) -> list[float]:
        terms = set(tokenize(query))
        out = []
        for tf, dl in zip(self.docs, self.lengths):
            s = 0.0
            for t in terms:
                f = tf.get(t)
                if not f:
                    continue
                norm = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                s += self.idf(t) * f * (self.k1 + 1) / norm
            out.append(s)
        return out

    def top_k(self, query: str, k: int) -> list[tuple[int, float]]:
        ranked = sorted(enumerate(self.scores(query)), key=lambda x: x[1], reverse=True)
        return [(i, s) for i, s in ranked[:k] if s > 0]
