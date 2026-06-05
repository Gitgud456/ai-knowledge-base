"""RAPTOR — Recursive Abstractive Processing for Tree-Organized Retrieval
(Sarthi et al., 2024).

Idea: cluster all chunks by their dense embedding, summarise each cluster
with an LLM, then re-cluster the summaries to form a higher level. Repeat
until you have one root summary. At query time the retriever can hit
*any* level — leaves give literal context, internal nodes give thematic
context ("what do my notes collectively say about X").

What we build here

  * :func:`build_tree` — Reads every chunk from the main Qdrant collection,
    clusters with UMAP→GMM (or just GMM if UMAP is unavailable), summarises
    each cluster with Ollama, embeds the summaries, and writes them as new
    chunks back into the main collection with ``source_type='raptor'`` and
    a ``level`` payload field. Levels >0 are RAPTOR summary nodes.
  * Retrieval already finds these alongside leaves; the only thing the
    retriever has to do differently is *prefer* the deepest matching level
    when summaries and leaves tie (handled by a small re-rank tweak).

Cost model

  * One LLM call per cluster per level. With ~50k chunks and avg fan-out
    of ~10 you end up summarising ~5k clusters → ~5500 LLM calls total
    across levels. At 1s/call locally that's ~1.5h, comparable to a single
    contextual-retrieval pass. Skippable on opt-in.

Failure modes

  * If clustering doesn't converge for a level, we stop early — the tree
    built so far is fully usable.
  * Failed summaries are dropped (no level-N node for that cluster).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import ollama

from akb.config import RaptorConfig, load_settings
from akb.embed.providers import EmbeddedBatch, SparseVec, get_embedder
from akb.obs.logging import get_logger
from akb.schemas import Chunk, SourceType
from akb.store.qdrant_store import COLLECTION, QdrantStore, _hydrate, get_store

log = get_logger(__name__)


SUMMARY_PROMPT = """You are summarising a cluster of related notes from a personal knowledge base.

Write a tight 3-6 sentence synthesis that:
  * Names the central theme the chunks share
  * Lists the load-bearing facts or claims (terse, no recap of "this discusses…")
  * Calls out tensions or open questions if any

Respond with ONLY the synthesis, no preamble.

CLUSTER CHUNKS:
{cluster}

SYNTHESIS:"""


@dataclass
class TreeStats:
    levels: int = 0
    summaries_written: int = 0
    failed_summaries: int = 0
    per_level: list[int] = field(default_factory=list)


def _fetch_all_leaves(store: QdrantStore) -> tuple[list[Chunk], list[list[float]]]:
    """Scroll every leaf chunk (level=0 or missing) and gather its dense vector."""
    from qdrant_client import models  # type: ignore[import-untyped]

    client = store.client
    chunks: list[Chunk] = []
    vectors: list[list[float]] = []
    offset: Any = None
    flt = models.Filter(
        must_not=[
            models.FieldCondition(key="source_type", match=models.MatchValue(value="raptor"))
        ]
    )
    while True:
        batch, offset = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=flt,
            with_payload=True,
            with_vectors=["dense"],
            limit=512,
            offset=offset,
        )
        for p in batch:
            vec = p.vector if isinstance(p.vector, list) else (p.vector or {}).get("dense")
            if not vec:
                continue
            chunks.append(_hydrate(p.payload or {}))
            vectors.append(list(vec))
        if offset is None:
            break
    return chunks, vectors


def _cluster_indices(
    vectors: list[list[float]],
    cfg: RaptorConfig,
    seed: int,
) -> list[int]:
    """Assign each vector to a cluster id. UMAP→GMM is preferred; pure GMM is
    the fallback. Returns parallel list of cluster ids."""
    import numpy as np

    arr = np.asarray(vectors, dtype=float)
    if len(arr) < cfg.min_cluster_size * 2:
        # Not enough points to form even two viable clusters → trivial singleton.
        return [0] * len(arr)
    k = max(2, min(cfg.max_clusters_per_level, len(arr) // cfg.min_cluster_size))
    if k < 2:
        return [0] * len(arr)

    reduced = arr
    try:
        import umap  # type: ignore[import-untyped]

        n_neighbors = min(cfg.umap_n_neighbors, len(arr) - 1)
        reducer = umap.UMAP(
            n_components=min(cfg.umap_dim, len(arr) - 1),
            n_neighbors=max(2, n_neighbors),
            metric="cosine",
            random_state=seed,
        )
        reduced = reducer.fit_transform(arr)
    except Exception as e:
        log.warning("raptor.umap.skip", error=str(e))

    try:
        from sklearn.mixture import GaussianMixture

        gmm = GaussianMixture(n_components=k, random_state=seed)
        gmm.fit(reduced)
        return gmm.predict(reduced).tolist()
    except Exception as e:
        log.warning("raptor.gmm.skip", error=str(e))
        return [i % k for i in range(len(arr))]


def _summary_text(chunks: list[Chunk]) -> str:
    return "\n\n---\n\n".join(
        f"[{c.metadata.get('title') or c.source_id}] {c.text}" for c in chunks
    )


def _summarise_cluster(chunks: list[Chunk], model: str) -> str:
    body = _summary_text(chunks)[:12000]
    prompt = SUMMARY_PROMPT.format(cluster=body)
    try:
        resp = ollama.generate(
            model=model,
            prompt=prompt,
            options={"temperature": 0.2, "num_predict": 400},
        )
        return str(resp.get("response", "")).strip()
    except Exception as e:
        log.warning("raptor.summary.error", error=str(e))
        return ""


def _make_summary_chunk(level: int, cluster_id: int, members: list[Chunk], text: str) -> Chunk:
    member_sources = sorted({c.source_id for c in members})
    h = hashlib.sha256("|".join(member_sources).encode()).hexdigest()[:12]
    source_id = f"raptor:L{level}:{h}"
    now = datetime.now(timezone.utc).isoformat()
    return Chunk(
        source_id=source_id,
        source_type=SourceType.raptor,
        text=text,
        chunk_index=cluster_id,
        metadata={
            "title": f"RAPTOR L{level} cluster {cluster_id}",
            "level": level,
            "member_sources": member_sources,
            "created_at": now,
            "modified_at": now,
        },
    )


def _embed_and_upsert(
    chunks: list[Chunk], store: QdrantStore
) -> tuple[int, list[list[float]]]:
    """Embed the summary chunks and upsert. Returns ``(n_upserted, dense_vecs)``
    so the caller can feed them straight into the next clustering pass."""
    if not chunks:
        return 0, []
    embedder = get_embedder()
    emb = embedder.embed_documents([c.text for c in chunks])
    if not emb.sparse:
        emb = EmbeddedBatch(
            dense=emb.dense,
            sparse=[SparseVec(indices=[], values=[]) for _ in chunks],
        )
    n = store.upsert(chunks, emb)
    return n, emb.dense


def build_tree(
    *,
    store: QdrantStore | None = None,
    cfg: RaptorConfig | None = None,
    seed: int = 42,
) -> TreeStats:
    """Build (or rebuild) the RAPTOR summary tree on top of the existing index."""
    settings = load_settings()
    cfg = cfg or settings.raptor
    store = store or get_store()
    stats = TreeStats()

    log.info("raptor.build.start")
    leaves, vectors = _fetch_all_leaves(store)
    if len(leaves) < cfg.min_cluster_size * 2:
        log.info("raptor.build.skip_too_small", n=len(leaves))
        return stats

    current_chunks = leaves
    current_vectors = vectors
    level = 1

    while level <= cfg.max_levels:
        if len(current_chunks) < cfg.min_cluster_size * 2:
            break
        labels = _cluster_indices(current_vectors, cfg, seed=seed + level)
        clusters: dict[int, list[Chunk]] = {}
        for ch, lab in zip(current_chunks, labels):
            clusters.setdefault(lab, []).append(ch)

        next_summaries: list[Chunk] = []
        for cid, members in sorted(clusters.items()):
            if len(members) < cfg.min_cluster_size:
                continue
            text = _summarise_cluster(members, cfg.summary_model or settings.llm.local_model)
            if not text:
                stats.failed_summaries += 1
                continue
            chunk = _make_summary_chunk(level, cid, members, text)
            chunk.metadata["level"] = level
            next_summaries.append(chunk)

        n_written, dense = _embed_and_upsert(next_summaries, store)
        stats.summaries_written += n_written
        stats.per_level.append(n_written)
        stats.levels = level
        log.info("raptor.level.done", level=level, summaries=n_written)

        if n_written < cfg.min_cluster_size * 2:
            break
        current_chunks = next_summaries
        current_vectors = dense
        level += 1

    log.info("raptor.build.done", **stats.__dict__)
    return stats


def delete_tree(store: QdrantStore | None = None) -> int:
    """Drop every RAPTOR summary (any level) from the main collection."""
    from qdrant_client import models  # type: ignore[import-untyped]

    store = store or get_store()
    flt = models.Filter(
        must=[
            models.FieldCondition(key="source_type", match=models.MatchValue(value="raptor"))
        ]
    )
    n = store.client.count(COLLECTION, count_filter=flt, exact=True).count
    store.client.delete(
        collection_name=COLLECTION,
        points_selector=models.FilterSelector(filter=flt),
        wait=True,
    )
    log.info("raptor.delete.done", removed=n)
    return int(n)
