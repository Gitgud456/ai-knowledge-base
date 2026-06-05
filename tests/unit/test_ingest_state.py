"""IngestState round-trip and diff semantics — small but high-value tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from akb.store.sqlite_state import IngestState


@pytest.fixture
def state(tmp_path: Path) -> IngestState:
    return IngestState(db_path=tmp_path / "ingest_state.db")


def test_upsert_then_get(state: IngestState) -> None:
    state.upsert_source("obsidian:a.md", "/v/a.md", "hash-1", ["c1", "c2"])
    rec = state.get("obsidian:a.md")
    assert rec is not None
    assert rec.content_hash == "hash-1"
    assert sorted(state.chunk_ids_for("obsidian:a.md")) == ["c1", "c2"]


def test_upsert_replaces_chunk_ids(state: IngestState) -> None:
    state.upsert_source("obsidian:a.md", "/v/a.md", "h1", ["c1", "c2"])
    state.upsert_source("obsidian:a.md", "/v/a.md", "h2", ["c3"])
    assert state.chunk_ids_for("obsidian:a.md") == ["c3"]
    assert state.get("obsidian:a.md").content_hash == "h2"  # type: ignore[union-attr]


def test_delete_cascades(state: IngestState) -> None:
    state.upsert_source("obsidian:a.md", "/v/a.md", "h", ["c1"])
    deleted = state.delete_source("obsidian:a.md")
    assert deleted == ["c1"]
    assert state.get("obsidian:a.md") is None
    assert state.chunk_ids_for("obsidian:a.md") == []


def test_all_source_ids(state: IngestState) -> None:
    state.upsert_source("obsidian:a.md", "/v/a.md", "h", [])
    state.upsert_source("obsidian:b.md", "/v/b.md", "h", [])
    assert state.all_source_ids() == {"obsidian:a.md", "obsidian:b.md"}
