from __future__ import annotations

from evodef.retrieval.bm25 import BM25Index
from evodef.retrieval.dense import DenseIndex
from evodef.retrieval.hybrid import rrf_fuse
from evodef.schemas import StatuteChunk


def _chunk(chunk_id: str, text: str) -> StatuteChunk:
    return StatuteChunk(chunk_id=chunk_id, section_id=chunk_id, text=text, source_file="f")


CHUNKS = [
    _chunk("c1", "qualifying relative gross income test dependent"),
    _chunk("c2", "qualifying child relationship residency age test"),
    _chunk("c3", "rate of tax imposed on employer wages"),
]


class TestBM25:
    def test_relevant_chunk_ranks_first(self):
        index = BM25Index.build(CHUNKS)
        results = index.search("qualifying relative gross income", top_k=3)
        assert results[0][0] == "c1"

    def test_deterministic_tie_break_by_chunk_id(self):
        chunks = [_chunk("b", "same same same"), _chunk("a", "same same same")]
        index = BM25Index.build(chunks)
        results = index.search("same", top_k=2)
        assert [r[0] for r in results] == ["a", "b"]

    def test_empty_corpus(self):
        index = BM25Index.build([])
        assert index.search("anything") == []


class TestDenseIndex:
    def test_cache_roundtrip(self, tmp_path):
        calls = {"n": 0}

        def embed_fn(texts):
            calls["n"] += 1
            return [[float(len(t)), 1.0, 0.0] for t in texts]

        cache_path = tmp_path / "dense.npy"
        idx1 = DenseIndex.build(CHUNKS, embed_fn, cache_path=cache_path)
        assert calls["n"] == 1
        idx2 = DenseIndex.build(CHUNKS, embed_fn, cache_path=cache_path)
        assert calls["n"] == 1  # cache hit, no re-embed
        assert idx1.chunk_ids == idx2.chunk_ids

    def test_search_returns_top_k_sorted(self):
        def embed_fn(texts):
            # c1 gets a vector identical to the query
            return [[1.0, 0.0] if "c1" in t else [0.0, 1.0] for t in texts]

        chunks = [_chunk("c1", "c1 marker text"), _chunk("c2", "other text")]
        idx = DenseIndex.build(chunks, embed_fn)
        results = idx.search([1.0, 0.0], top_k=1)
        assert results[0][0] == "c1"


class TestRRF:
    def test_fuses_multiple_lists(self):
        list_a = [("c1", 5.0), ("c2", 3.0), ("c3", 1.0)]
        list_b = [("c2", 0.9), ("c3", 0.5), ("c1", 0.2)]
        fused = rrf_fuse([list_a, list_b], k_rrf=60)
        ids = [f[0] for f in fused]
        assert set(ids) == {"c1", "c2", "c3"}
        # c2 is rank 2 in both lists -> should not be last
        assert ids.index("c2") < 2

    def test_top_k_truncates(self):
        list_a = [("c1", 1.0), ("c2", 1.0), ("c3", 1.0)]
        fused = rrf_fuse([list_a], top_k=2)
        assert len(fused) == 2

    def test_deterministic_tie_break(self):
        # Genuine RRF tie: "a" and "b" each hold rank 1 in one list apiece,
        # so they end up with equal fused scores.
        fused = rrf_fuse([[("b", 1.0)], [("a", 1.0)]])
        assert fused[0][1] == fused[1][1]
        assert fused[0][0] == "a"
