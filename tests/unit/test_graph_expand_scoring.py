"""Graph-expanded chunks must carry a non-trivial ``rrf_score`` so they survive
the ``pool = candidates[:reranker_top_n]`` slice downstream."""

from __future__ import annotations

from typing import Any

import pytest

from akb.config import RetrieveConfig
from akb.ingest.graph import VaultGraph
from akb.retrieve.graph_expand import expand
from akb.schemas import Chunk, RetrievedChunk, SourceType


def _rc(cid: str, sid: str, rrf: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(chunk_id=cid, source_id=sid, source_type=SourceType.obsidian, text=cid),
        rrf_score=rrf,
    )


class _FakeStore:
    def __init__(self, chunks_per_source: dict[str, list[Chunk]]) -> None:
        self._m = chunks_per_source

    def fetch_chunks_for_sources(self, sources: Any, limit_per_source: int | None = None) -> list[Chunk]:
        out: list[Chunk] = []
        for s in sources:
            out.extend(self._m.get(s, [])[: (limit_per_source or 100)])
        return out


def test_graph_chunks_get_baseline_score(monkeypatch: pytest.MonkeyPatch) -> None:
    g = VaultGraph(
        forward={"obsidian:A.md": {"obsidian:B.md"}, "obsidian:B.md": set()},
        backward={"obsidian:B.md": {"obsidian:A.md"}, "obsidian:A.md": set()},
    )
    seeds = [_rc("a0", "obsidian:A.md", 1.0)]
    extras_per_source = {
        "obsidian:B.md": [
            Chunk(chunk_id="b0", source_id="obsidian:B.md", source_type=SourceType.obsidian, text="b0"),
        ],
    }
    fake_store: Any = _FakeStore(extras_per_source)

    cfg = RetrieveConfig(graph_expand=True, graph_hops=1, graph_expand_limit=5)
    out = expand(seeds, g, cfg=cfg, store=fake_store)
    out_ids = [rc.chunk.chunk_id for rc in out]
    assert "b0" in out_ids
    # Sorted by rrf_score desc -> seed first, then graph chunk
    assert out[0].chunk.chunk_id == "a0"
    b = next(rc for rc in out if rc.chunk.chunk_id == "b0")
    assert b.rrf_score is not None and b.rrf_score > 0.0
    # Decay applied: graph chunk score < seed score
    assert b.rrf_score < (seeds[0].rrf_score or 1.0)


def test_graph_expand_no_op_when_disabled() -> None:
    g = VaultGraph()
    seeds = [_rc("a0", "obsidian:A.md", 1.0)]
    cfg = RetrieveConfig(graph_expand=False)
    out = expand(seeds, g, cfg=cfg, store=None)
    assert out == seeds


def test_graph_expand_returns_seeds_when_no_neighbours() -> None:
    g = VaultGraph(forward={"obsidian:A.md": set()}, backward={"obsidian:A.md": set()})
    seeds = [_rc("a0", "obsidian:A.md", 1.0)]
    cfg = RetrieveConfig(graph_expand=True, graph_hops=1)
    out = expand(seeds, g, cfg=cfg, store=None)
    assert out == seeds
