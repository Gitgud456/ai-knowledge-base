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
from akb.ingest.scrubber import scrub_chunks
from akb.ingest.upsert import upsert_chunks
from akb.obs.logging import get_logger
from akb.schemas import Document
from akb.store.qdrant_store import QdrantStore, get_store
from akb.store.sqlite_state import IngestState

log = get_logger(__name__)


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
    *,
    restrict_paths: list[Path] | None = None,
) -> SyncPlan:
    """Compute a sync plan.

    Default (``restrict_paths=None``) walks the whole vault and produces a
    full add/change/delete plan. Pass ``restrict_paths`` to limit add/change
    inspection to just those paths — the watcher uses this so a single
    save doesn't trigger a 50k-file rescan. In restricted mode, deletes are
    inferred only for the passed paths (i.e. a passed path that no longer
    exists on disk). For a full delete sweep, run with ``restrict_paths=None``.
    """
    settings = load_settings()
    vault = vault or settings.paths.vault
    state = state or IngestState()
    skip = {d.lower() for d in settings.ingest.skip_dirs}

    if restrict_paths is None:
        candidates = _walk_vault(vault, skip)
        full_sweep = True
    else:
        candidates = [p for p in restrict_paths if p.suffix.lower() == ".md"]
        full_sweep = False
    index = _build_index(vault, settings.ingest)

    plan = SyncPlan()
    seen_source_ids: set[str] = set()

    for path in candidates:
        if not path.exists():
            # In restricted mode, a passed path that's gone becomes a delete.
            rel = path.relative_to(vault) if path.is_relative_to(vault) else path
            plan.deleted.append(f"obsidian:{rel.as_posix()}")
            continue
        rel = path.relative_to(vault) if path.is_relative_to(vault) else path
        source_id = f"obsidian:{rel.as_posix()}"
        seen_source_ids.add(source_id)
        existing = state.get(source_id)
        if existing is None:
            plan.added.append(path)
            continue
        new_hash = _doc_for_path(path, vault, index).content_hash
        if new_hash != existing.content_hash:
            plan.changed.append(path)
        else:
            state.touch(source_id)

    if full_sweep:
        known = state.all_source_ids()
        plan.deleted = sorted(known - seen_source_ids)
    else:
        plan.deleted = sorted(set(plan.deleted))
    log.info(
        "sync.plan",
        added=len(plan.added),
        changed=len(plan.changed),
        deleted=len(plan.deleted),
        mode="full" if full_sweep else "restricted",
    )
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
        log.info("sync.delete", source_id=sid)

    for path in [*plan.added, *plan.changed]:
        rel = path.relative_to(vault) if path.is_relative_to(vault) else path
        source_id = f"obsidian:{rel.as_posix()}"

        # Drop any prior chunks for this source from Qdrant first (handles "changed").
        store.delete_by_source(source_id)

        doc = _doc_for_path(path, vault, index)
        chunks = chunk_document(doc)
        chunks = scrub_chunks(chunks)
        if ctx is not None:
            chunks = ctx.contextualize(doc, chunks)
        chunks = dedupe_chunks(chunks)
        if not chunks:
            # Source produced zero chunks (empty note, all-frontmatter, deduped
            # against itself). Record the hash anyway so the next plan_sync
            # doesn't keep re-enqueuing this file as "changed".
            state.upsert_source(
                source_id=doc.source_id,
                path=str(path),
                content_hash=doc.content_hash,
                chunk_ids=[],
            )
            continue

        n = upsert_chunks(chunks, store=store)
        state.upsert_source(
            source_id=doc.source_id,
            path=str(path),
            content_hash=doc.content_hash,
            chunk_ids=[c.chunk_id for c in chunks],
        )
        upserts += n
        log.info("sync.upsert", source_id=doc.source_id, chunks=n)
        if callable(on_progress):
            on_progress(path, n)  # type: ignore[misc]

    log.info("sync.done", upserts=upserts, deletes=deletes)
    return {"upserts": upserts, "deletes": deletes}


def run_sync_once(vault: Path | None = None) -> dict[str, int]:
    plan = plan_sync(vault=vault)
    return apply_sync(plan, vault=vault)


_ = Iterable  # re-exported for typing convenience
