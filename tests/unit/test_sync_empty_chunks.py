"""Empty-chunks-after-dedupe must record the source hash anyway, otherwise
``plan_sync`` re-enqueues the file as "changed" on every run."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from akb.store.sqlite_state import IngestState


class _FakeStore:
    def __init__(self) -> None:
        self.deletes: list[str] = []
        self.upserts: list[tuple[str, int]] = []

    def delete_by_source(self, sid: str) -> int:
        self.deletes.append(sid)
        return 0


def test_empty_doc_records_hash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A note that yields zero chunks after dedupe must still update ingest_state."""
    note = tmp_path / "empty.md"
    note.write_text("", encoding="utf-8")
    state = IngestState(db_path=tmp_path / "state.db")

    from akb.ingest import sync as sync_mod
    from akb.schemas import Document, SourceType

    # Force a fixed Document so we control the hash regardless of loader quirks.
    fixed_doc = Document(
        source_id="obsidian:empty.md",
        source_type=SourceType.obsidian,
        title="empty",
        content="(empty)",
    )

    monkeypatch.setattr(sync_mod, "_doc_for_path", lambda p, v, i: fixed_doc)
    monkeypatch.setattr(sync_mod, "chunk_document", lambda d: [])
    monkeypatch.setattr(sync_mod, "dedupe_chunks", lambda xs: list(xs))
    monkeypatch.setattr(sync_mod, "_build_index", lambda v, c: {})

    # Skip the contextualizer entirely
    from akb.config import IngestConfig, load_settings, reset_settings_cache

    reset_settings_cache()
    real = load_settings()
    monkeypatch.setattr(
        real.ingest,
        "contextual_retrieval",
        False,
        raising=False,
    )

    plan = sync_mod.SyncPlan(added=[note])
    fake_store: Any = _FakeStore()
    result = sync_mod.apply_sync(plan, vault=tmp_path, state=state, store=fake_store)

    assert result["upserts"] == 0
    rec = state.get("obsidian:empty.md")
    assert rec is not None, "empty-chunks source must be recorded in ingest_state"
    assert rec.content_hash == fixed_doc.content_hash
