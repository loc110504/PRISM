"""Dense retrieval: Ollama embeddings cached to .npy, cosine similarity search.

(03_IMPLEMENTATION_SPEC.md #4 "Dense": "embed chunk text once and cache to
`.npy`, normalize vectors, cosine similarity". FAISS is optional; the SARA
corpus is small enough that a plain NumPy matmul is sufficient.)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from ..schemas import StatuteChunk
from ..utils import sha256_json

EmbedFn = Callable[[list[str]], list[list[float]]]


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


@dataclass
class DenseIndex:
    chunk_ids: list[str] = field(default_factory=list)
    vectors: np.ndarray | None = None  # (N, D), L2-normalized

    @classmethod
    def build(
        cls,
        chunks: list[StatuteChunk],
        embed_fn: EmbedFn,
        cache_path: str | Path | None = None,
        cache_key: str | None = None,
        force_rebuild: bool = False,
    ) -> "DenseIndex":
        """Embed each chunk once; reuse `cache_path` on subsequent runs.

        The cache is keyed on the complete chunk content and embedding-model
        tag supplied by the caller. Checking IDs alone is insufficient: an
        edited statute can retain its ID while requiring new embeddings.
        """
        chunk_ids = [c.chunk_id for c in chunks]
        corpus_hash = sha256_json([
            {"chunk_id": c.chunk_id, "text": c.text} for c in chunks
        ])
        cache_path = Path(cache_path) if cache_path is not None else None

        if cache_path is not None and cache_path.exists() and not force_rebuild:
            cached = np.load(cache_path, allow_pickle=True).item()
            if (
                cached.get("chunk_ids") == chunk_ids
                and cached.get("corpus_hash") == corpus_hash
                and cached.get("cache_key") == cache_key
            ):
                return cls(chunk_ids=chunk_ids, vectors=cached["vectors"])

        raw = embed_fn([c.text for c in chunks])
        vectors = _normalize(np.array(raw, dtype=np.float32))

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(
                cache_path,
                {
                    "chunk_ids": chunk_ids,
                    "corpus_hash": corpus_hash,
                    "cache_key": cache_key,
                    "vectors": vectors,
                },
                allow_pickle=True,
            )

        return cls(chunk_ids=chunk_ids, vectors=vectors)

    def search(self, query_embedding: list[float], top_k: int = 20) -> list[tuple[str, float]]:
        if self.vectors is None or len(self.chunk_ids) == 0:
            return []
        q = _normalize(np.array([query_embedding], dtype=np.float32))[0]
        scores = self.vectors @ q
        ranked = sorted(
            zip(self.chunk_ids, scores.tolist()),
            key=lambda x: (-x[1], x[0]),
        )
        return ranked[:top_k]
