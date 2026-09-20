"""Sparse BM25 retrieval over the statute corpus (03_IMPLEMENTATION_SPEC.md #4)."""
from __future__ import annotations

from dataclasses import dataclass, field

from rank_bm25 import BM25Okapi

from ..schemas import StatuteChunk
from ..utils import tokenize


@dataclass
class BM25Index:
    chunk_ids: list[str] = field(default_factory=list)
    _bm25: BM25Okapi | None = None

    @classmethod
    def build(cls, chunks: list[StatuteChunk]) -> "BM25Index":
        index = cls(chunk_ids=[c.chunk_id for c in chunks])
        if not chunks:
            return index  # rank_bm25 divides by zero on an empty corpus
        tokenized_corpus = [tokenize(c.text) for c in chunks]
        index._bm25 = BM25Okapi(tokenized_corpus)
        return index

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        if self._bm25 is None or not self.chunk_ids:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(
            zip(self.chunk_ids, scores),
            key=lambda x: (-x[1], x[0]),  # deterministic tie-break by chunk_id
        )
        return ranked[:top_k]
