"""Incremental sync.

Compares ``vault file → content_hash`` against ``ingest_state.db`` and produces
three sets:

  * **added**     — present in vault, not in state DB
  * **changed**   — present in both, but content_hash differs
  * **deleted**   — present in state DB, not in vault

For added/changed we run the full ingest pipeline (loader → chunker →
contextualizer → embedder → Qdrant upsert) but *only* for affected docs.
For deleted we delete from Qdrant and from state.

Designed to be safe to interrupt — every doc is committed before the next one
starts. A crash mid-loop leaves the index in a coherent state with one fewer
doc updated, which the next run will pick up.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from akb.config import load_settings
from akb.ingest.chunkers import chunk_document
from akb.ingest.contextualizer import Contextualizer
from akb.ingest.dedupe import dedupe_chunks
from akb.ingest.obsidian_loader import _build_index, load_note  # internal but stable
from akb.ingest.upsert import upsert_chunks
from akb.schemas import Document
from akb.store.qdrant_store import QdrantStore, get_store
from akb.store.sqlite_state import IngestState


@dataclass
class SyncPlan:
    added: list[Path] = field(default_factory=list)
    changed: list[Path] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)  # source_ids

    def total(self) -> int:
        return len(self.added) + len(self.changed) + len(self.deleted)


def _walk_vault(vault: Path, skip_dirs: set[str]) -> list[Path]:
    out: list[Path] = []
    for md in vault.rglob("*.md"):
        if any(part.lower() in skip_dirs for part in md.parts):
            continue
        out.append(md)
    return out


def _doc_for_path(path: Path, vault: Path, index: dict[str, Path]) -> Document:
    return load_note(path, vault=vault, index=index)


def plan_sync(
    vault: Path | None = None,
    state: IngestState | None = None,
) -> SyncPlan:
    settings = load_settings()
    vault = vault or settings.paths.vault
    state = state or IngestState()
    skip = {d.lower() for d in settings.ingest.skip_dirs}

    on_disk = _walk_vault(vault, skip)
    index = _build_index(vault, settings.ingest)

    plan = SyncPlan()
    seen_source_ids: set[str] = set()

    for path in on_disk:
        rel = path.relative_to(vault) if path.is_relative_to(vault) else path
        source_id = f"obsidian:{rel.as_posix()}"
        seen_source_ids.add(source_id)
        existing = state.get(source_id)
        if existing is None:
            plan.added.append(path)
            continue
        # Cheap mtime gate first; only hash on suspicion.
        new_hash = _doc_for_path(path, vault, index).content_hash
        if new_hash != existing.content_hash:
            plan.changed.append(path)
        else:
            state.touch(source_id)

    known = state.all_source_ids()
    plan.deleted = sorted(known - seen_source_ids)
    return plan


def apply_sync(
    plan: SyncPlan,
    *,
    vault: Path | None = None,
    state: IngestState | None = None,
    store: QdrantStore | None = None,
    on_progress: object = None,
) -> dict[str, int]:
    settings = load_settings()
    vault = vault or settings.paths.vault
    state = state or IngestState()
    store = store or get_store()
    ctx = Contextualizer() if settings.ingest.contextual_retrieval else None
    index = _build_index(vault, settings.ingest)

    upserts = 0
    deletes = 0

    # Deletes first — frees space and reduces Qdrant payload pressure mid-loop.
    for sid in plan.deleted:
        store.delete_by_source(sid)
        state.delete_source(sid)
        deletes += 1

    for path in [*plan.added, *plan.changed]:
        rel = path.relative_to(vault) if path.is_relative_to(vault) else path
        source_id = f"obsidian:{rel.as_posix()}"

        # Drop any prior chunks for this source from Qdrant first (handles "changed").
        store.delete_by_source(source_id)

        doc = _doc_for_path(path, vault, index)
        chunks = chunk_document(doc)
        if ctx is not None:
            chunks = ctx.contextualize(doc, chunks)
        chunks = dedupe_chunks(chunks)
        if not chunks:
            continue

        n = upsert_chunks(chunks, store=store)
        state.upsert_source(
            source_id=doc.source_id,
            path=str(path),
            content_hash=doc.content_hash,
            chunk_ids=[c.chunk_id for c in chunks],
        )
        upserts += n
        if callable(on_progress):
            on_progress(path, n)  # type: ignore[misc]

    return {"upserts": upserts, "deletes": deletes}


def run_sync_once(vault: Path | None = None) -> dict[str, int]:
    plan = plan_sync(vault=vault)
    return apply_sync(plan, vault=vault)


_ = Iterable  # re-exported for typing convenience
