"""Regression for the most dangerous bug we found in review: the original
``Chunk.chunk_id`` was a random uuid4, so re-creating "the same chunk" produced
a different id, which meant ``akb ingest`` silently duplicated every point on a
re-run. The deterministic id rule must hold:

    chunk_id = f"{source_id}::{chunk_index}::{sha256(text)[:16]}"

and the Qdrant point id derived from it must also be stable."""

from __future__ import annotations

from akb.schemas import Chunk, SourceType
from akb.store.qdrant_store import _point_id


def _c(text: str, idx: int = 0, sid: str = "obsidian:a.md") -> Chunk:
    return Chunk(source_id=sid, source_type=SourceType.obsidian, text=text, chunk_index=idx)


def test_chunk_id_is_deterministic_across_instances() -> None:
    a = _c("hello world")
    b = _c("hello world")
    assert a.chunk_id == b.chunk_id


def test_chunk_id_changes_with_text() -> None:
    assert _c("hello").chunk_id != _c("world").chunk_id


def test_chunk_id_changes_with_index() -> None:
    assert _c("x", 0).chunk_id != _c("x", 1).chunk_id


def test_chunk_id_changes_with_source() -> None:
    assert _c("x", sid="A").chunk_id != _c("x", sid="B").chunk_id


def test_point_id_stable_for_same_chunk() -> None:
    assert _point_id(_c("hello").chunk_id) == _point_id(_c("hello").chunk_id)


def test_explicit_chunk_id_overrides_derivation() -> None:
    c = Chunk(
        chunk_id="my-explicit-id",
        source_id="t",
        source_type=SourceType.txt,
        text="anything",
    )
    assert c.chunk_id == "my-explicit-id"
