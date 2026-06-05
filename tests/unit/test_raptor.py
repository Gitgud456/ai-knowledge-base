"""RAPTOR: clustering + summary-chunk construction."""

from __future__ import annotations

from typing import Any

import pytest

from akb.config import RaptorConfig
from akb.ingest import raptor as rp
from akb.schemas import Chunk, SourceType


def _ch(text: str, sid: str, idx: int = 0) -> Chunk:
    return Chunk(source_id=sid, source_type=SourceType.obsidian, text=text, chunk_index=idx)


def test_make_summary_chunk_stable_id() -> None:
    members = [_ch("a", "obsidian:A.md"), _ch("b", "obsidian:B.md")]
    c1 = rp._make_summary_chunk(level=1, cluster_id=0, members=members, text="summary")
    c2 = rp._make_summary_chunk(level=1, cluster_id=0, members=members, text="summary")
    assert c1.source_id == c2.source_id
    assert c1.source_type == SourceType.raptor
    assert c1.metadata["level"] == 1
    assert c1.metadata["member_sources"] == ["obsidian:A.md", "obsidian:B.md"]


def test_cluster_indices_returns_one_per_vector() -> None:
    vectors = [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9], [0.5, 0.5]] * 4
    cfg = RaptorConfig(min_cluster_size=2, max_clusters_per_level=4)
    labels = rp._cluster_indices(vectors, cfg, seed=1)
    assert len(labels) == len(vectors)


def test_cluster_indices_single_cluster_on_too_few_vectors() -> None:
    cfg = RaptorConfig(min_cluster_size=10, max_clusters_per_level=8)
    labels = rp._cluster_indices([[1.0], [2.0]], cfg, seed=1)
    assert set(labels) == {0}


def test_summarise_cluster_returns_blank_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(**kwargs: Any) -> dict[str, str]:
        raise RuntimeError("offline")

    monkeypatch.setattr(rp.ollama, "generate", _boom)
    out = rp._summarise_cluster([_ch("x", "s")], model="m")
    assert out == ""
