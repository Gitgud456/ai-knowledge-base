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
    """Append 1-hop wikilink neighbours to ``seeds`` and re-sort by score.

    Subtle correctness point: the reranker downstream takes only the first
    ``reranker_top_n`` candidates. If we just append graph chunks at the tail,
    they get sliced off and graph expansion becomes inert whenever
    ``n_results ≥ reranker_top_n``. We instead assign each graph chunk a
    baseline ``rrf_score`` derived from its best seed (a small decay factor),
    then sort the merged list before returning — so graph chunks rank into
    the pool by score, not insertion order.
    """
    cfg = cfg or load_settings().retrieve
    if not cfg.graph_expand or graph is None or not seeds:
        return seeds

    store = store or get_store()
    # Best score per seed source for the decay attribution
    score_by_source: dict[str, float] = {}
    for rc in seeds:
        prev = score_by_source.get(rc.chunk.source_id, 0.0)
        cur = rc.rrf_score or 0.0
        if cur > prev:
            score_by_source[rc.chunk.source_id] = cur

    seed_sources = set(score_by_source)
    neighbour_to_seed: dict[str, str] = {}
    for sid in seed_sources:
        for n in graph.neighbours(sid, hops=cfg.graph_hops):
            if n in seed_sources or n in neighbour_to_seed:
                continue
            neighbour_to_seed[n] = sid
    if not neighbour_to_seed:
        return seeds

    extras = store.fetch_chunks_for_sources(
        neighbour_to_seed.keys(), limit_per_source=cfg.graph_expand_limit
    )
    seed_ids = {rc.chunk.chunk_id for rc in seeds}
    decay = 0.5  # graph chunks compete at half the seed's score
    appended: list[RetrievedChunk] = list(seeds)
    for c in extras:
        if c.chunk_id in seed_ids:
            continue
        seed_sid = neighbour_to_seed.get(c.source_id, "")
        baseline = score_by_source.get(seed_sid, 0.0) * decay
        appended.append(
            RetrievedChunk(chunk=c, rrf_score=baseline, expanded_from=seed_sid or "graph")
        )

    appended.sort(key=lambda rc: rc.rrf_score or 0.0, reverse=True)
    return appended
