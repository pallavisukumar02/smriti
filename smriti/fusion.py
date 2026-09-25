"""Reciprocal Rank Fusion: merge ranked lists from different retrievers."""

from __future__ import annotations


def reciprocal_rank_fusion(
    rankings: list[list[int]], k: int = 60, weights: list[float] | None = None
) -> list[tuple[int, float]]:
    """Combine ranked lists of ids. score(d) = sum_i w_i / (k + rank_i(d)).

    RRF uses only ranks, so dense and sparse scores never need to be put on the same scale.
    (Cormack, Clarke & Buettcher, 2009)
    """
    weights = weights or [1.0] * len(rankings)
    scores: dict[int, float] = {}
    for w, ranking in zip(weights, rankings):
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + w / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
