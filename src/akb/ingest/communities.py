"""Lightweight Graph-RAG via wikilink communities.

We already build a real wikilink graph at ingest. Microsoft GraphRAG's
expensive entity-extraction step is overkill here — the user wrote the
edges themselves. So this module:

  1. Loads the vault as Documents.
  2. Builds the wikilink graph (we already have :func:`build_graph`).
  3. Runs **Louvain** community detection on the undirected projection.
  4. For each community above ``min_community_size``, pulls a few representative
     chunks from each member note, asks the LLM for a thematic synthesis, and
     upserts that as a single ``source_type='community'`` chunk into the main
     Qdrant collection.

At query time these community summaries surface naturally for "global" /
thematic questions ("what are my notes about distributed systems") because
they cover the theme rather than one paragraph.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone

import networkx as nx
import ollama

from akb.config import CommunitiesConfig, load_settings
from akb.embed.providers import EmbeddedBatch, SparseVec, get_embedder
from akb.ingest.graph import VaultGraph
from akb.ingest.obsidian_loader import iter_vault
from akb.obs.logging import get_logger
from akb.schemas import Chunk, SourceType
from akb.store.qdrant_store import QdrantStore, get_store

log = get_logger(__name__)


COMMUNITY_PROMPT = """Below are short excerpts from a group of related notes in a personal knowledge base. They link to each other in the user's vault, so they form a thematic cluster.

Write a 4-7 sentence thematic synthesis that:
  * Names the unifying theme(s)
  * Highlights the most important shared facts / claims
  * Calls out divergences or tensions if any
  * Suggests follow-up questions a reader of this cluster might explore

Respond with ONLY the synthesis.

EXCERPTS:
{excerpts}

SYNTHESIS:"""


@dataclass
class CommunityStats:
    communities_found: int = 0
    summarised: int = 0
    failed: int = 0
    per_community_size: list[int] = field(default_factory=list)


def _louvain(graph: VaultGraph, cfg: CommunitiesConfig) -> dict[int, list[str]]:
    """Run Louvain on the undirected projection of the vault graph."""
    g = graph.to_networkx().to_undirected()
    if g.number_of_nodes() == 0:
        return {}
    # NetworkX 3.x has community.louvain_communities
    try:
        communities = list(
            nx.community.louvain_communities(g, resolution=cfg.resolution, seed=42)
        )
    except Exception as e:
        log.warning("communities.louvain.error", error=str(e))
        # Fallback: connected components
        communities = [set(c) for c in nx.connected_components(g)]

    out: dict[int, list[str]] = {}
    sized = sorted((c for c in communities if len(c) >= cfg.min_community_size),
                   key=len, reverse=True)
    for i, c in enumerate(sized[: cfg.max_communities]):
        out[i] = sorted(c)
    return out


def _excerpts_for_sources(
    store: QdrantStore,
    sources: list[str],
    per_source: int = 2,
    max_chars: int = 12000,
) -> str:
    chunks = store.fetch_chunks_for_sources(sources, limit_per_source=per_source)
    parts: list[str] = []
    used = 0
    for c in chunks:
        title = c.metadata.get("title") or c.source_id
        snippet = c.text.strip().replace("\n", " ")[:600]
        block = f"[{title}] {snippet}"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)


def _summarise(excerpts: str, model: str) -> str:
    if not excerpts:
        return ""
    try:
        resp = ollama.generate(
            model=model,
            prompt=COMMUNITY_PROMPT.format(excerpts=excerpts),
            options={"temperature": 0.2, "num_predict": 500},
        )
        return str(resp.get("response", "")).strip()
    except Exception as e:
        log.warning("communities.summary.error", error=str(e))
        return ""


def _make_chunk(community_id: int, members: list[str], text: str) -> Chunk:
    h = hashlib.sha256("|".join(members).encode()).hexdigest()[:12]
    source_id = f"community:{h}"
    now = datetime.now(timezone.utc).isoformat()
    return Chunk(
        source_id=source_id,
        source_type=SourceType.community,
        text=text,
        chunk_index=community_id,
        metadata={
            "title": f"Community {community_id} ({len(members)} notes)",
            "level": 1,
            "member_sources": members,
            "created_at": now,
            "modified_at": now,
        },
    )


def _upsert(chunks: list[Chunk], store: QdrantStore) -> int:
    if not chunks:
        return 0
    embedder = get_embedder()
    emb = embedder.embed_documents([c.text for c in chunks])
    if not emb.sparse:
        emb = EmbeddedBatch(
            dense=emb.dense,
            sparse=[SparseVec(indices=[], values=[]) for _ in chunks],
        )
    return store.upsert(chunks, emb)


def build_communities(
    *,
    store: QdrantStore | None = None,
    cfg: CommunitiesConfig | None = None,
) -> CommunityStats:
    settings = load_settings()
    cfg = cfg or settings.communities
    store = store or get_store()
    stats = CommunityStats()

    log.info("communities.build.start")
    from akb.ingest.graph import build_graph

    docs = list(iter_vault())
    if not docs:
        return stats
    graph = build_graph(docs)
    found = _louvain(graph, cfg)
    stats.communities_found = len(found)
    model = cfg.summary_model or settings.llm.local_model

    chunks: list[Chunk] = []
    for cid, members in found.items():
        excerpts = _excerpts_for_sources(store, members)
        text = _summarise(excerpts, model)
        if not text:
            stats.failed += 1
            continue
        chunks.append(_make_chunk(cid, members, text))
        stats.per_community_size.append(len(members))

    n = _upsert(chunks, store)
    stats.summarised = n
    log.info("communities.build.done", **stats.__dict__)
    return stats


def delete_communities(store: QdrantStore | None = None) -> int:
    from qdrant_client import models  # type: ignore[import-untyped]

    from akb.store.qdrant_store import COLLECTION

    store = store or get_store()
    flt = models.Filter(
        must=[
            models.FieldCondition(
                key="source_type", match=models.MatchValue(value="community")
            )
        ]
    )
    n = store.client.count(COLLECTION, count_filter=flt, exact=True).count
    store.client.delete(
        collection_name=COLLECTION,
        points_selector=models.FilterSelector(filter=flt),
        wait=True,
    )
    log.info("communities.delete.done", removed=n)
    return int(n)
