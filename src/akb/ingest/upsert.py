"""Embed + upsert orchestration.

Used by both the CLI (`akb ingest`, `akb reindex`) and the future watchfiles
sync loop (Phase 6). Batches embedding calls to keep BGE-M3 memory bounded.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from akb.config import IngestConfig, load_settings
from akb.embed.providers import EmbeddedBatch, SparseVec, get_embedder
from akb.schemas import Chunk
from akb.store.qdrant_store import QdrantStore, get_store


def _batched(items: list[Chunk], n: int) -> Iterator[list[Chunk]]:
    for i in range(0, len(items), n):
        yield items[i : i + n]


def upsert_chunks(
    chunks: Iterable[Chunk],
    *,
    store: QdrantStore | None = None,
    cfg: IngestConfig | None = None,
) -> int:
    settings = load_settings()
    cfg = cfg or settings.ingest
    store = store or get_store()
    embedder = get_embedder()

    materialised = list(chunks)
    total = 0
    for batch in _batched(materialised, cfg.batch_size):
        texts = [c.embed_text for c in batch]
        emb = embedder.embed_documents(texts)
        # Pad sparse list if disabled, so QdrantStore.upsert can iterate safely.
        if not emb.sparse:
            emb = EmbeddedBatch(
                dense=emb.dense,
                sparse=[SparseVec(indices=[], values=[]) for _ in batch],
            )
        total += store.upsert(batch, emb)
    return total
