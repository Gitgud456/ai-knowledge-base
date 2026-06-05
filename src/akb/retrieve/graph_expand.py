"""1-hop wikilink graph expansion.

After hybrid retrieval but *before* the reranker, pull in chunks from notes that
are wikilink-linked to any of the current top hits. The reranker then picks the
final top-k from the union.

Why this works: in Obsidian, the explicit link graph encodes "I considered these
together when writing" — a strong signal the standard embedding doesn't see.
"""

from __future__ import annotations

from akb.config import RetrieveConfig, load_settings
from akb.ingest.graph import VaultGraph
from akb.schemas import RetrievedChunk
from akb.store.qdrant_store import QdrantStore, get_store


def expand(
    seeds: list[RetrievedChunk],
    graph: VaultGraph | None,
    *,
    cfg: RetrieveConfig | None = None,
    store: QdrantStore | None = None,
) -> list[RetrievedChunk]:
    cfg = cfg or load_settings().retrieve
    if not cfg.graph_expand or graph is None or not seeds:
        return seeds

    store = store or get_store()
    seed_sources = {rc.chunk.source_id for rc in seeds}
    neighbour_sources: set[str] = set()
    for sid in seed_sources:
        neighbour_sources |= graph.neighbours(sid, hops=cfg.graph_hops)
    neighbour_sources -= seed_sources
    if not neighbour_sources:
        return seeds

    extras = store.fetch_chunks_for_sources(
        neighbour_sources, limit_per_source=cfg.graph_expand_limit
    )
    seed_ids = {rc.chunk.chunk_id for rc in seeds}
    appended: list[RetrievedChunk] = list(seeds)
    for c in extras:
        if c.chunk_id in seed_ids:
            continue
        appended.append(RetrievedChunk(chunk=c, expanded_from="graph"))
    return appended
