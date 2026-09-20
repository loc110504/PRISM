"""Reciprocal Rank Fusion (03_IMPLEMENTATION_SPEC.md #4 "Hybrid").

score(d) = sum_r 1 / (k_rrf + rank_r(d)), rank_r starting at 1.
"""
from __future__ import annotations


def rrf_fuse(
    ranked_lists: list[list[tuple[str, float]]],
    k_rrf: int = 60,
    top_k: int | None = None,
) -> list[tuple[str, float]]:
    """Fuse multiple (doc_id, score) ranked lists into one RRF-scored list.

    Only rank position matters, not the raw scores, so BM25 and cosine
    similarity scores (different scales) can be fused directly.
    """
    fused: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, (doc_id, _score) in enumerate(ranked, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k_rrf + rank)

    ordered = sorted(fused.items(), key=lambda x: (-x[1], x[0]))
    return ordered[:top_k] if top_k is not None else ordered
