"""RRF merge logic (the only piece of hybrid retrieval that doesn't need a live store)."""

from __future__ import annotations

from akb.retrieve.hybrid import _rrf_merge
from akb.schemas import Chunk, RetrievedChunk, SourceType


def _rc(cid: str, score: float = 0.0) -> RetrievedChunk:
    chunk = Chunk(chunk_id=cid, source_id="s", source_type=SourceType.txt, text="t")
    return RetrievedChunk(chunk=chunk, rrf_score=score)


def test_rrf_combines_consistent_top_ranks() -> None:
    a = [_rc("a", 1.0), _rc("b", 0.9), _rc("c", 0.8)]
    b = [_rc("b", 0.95), _rc("a", 0.7), _rc("d", 0.5)]
    out = _rrf_merge([a, b], k=60)
    # 'a' and 'b' show up in both lists in top-2 → should win
    top_ids = [rc.chunk.chunk_id for rc in out[:2]]
    assert set(top_ids) == {"a", "b"}
    assert "c" in [rc.chunk.chunk_id for rc in out]
    assert "d" in [rc.chunk.chunk_id for rc in out]


def test_rrf_score_strictly_decreasing() -> None:
    a = [_rc("a"), _rc("b"), _rc("c")]
    out = _rrf_merge([a], k=60)
    scores = [rc.rrf_score or 0 for rc in out]
    assert scores == sorted(scores, reverse=True)
