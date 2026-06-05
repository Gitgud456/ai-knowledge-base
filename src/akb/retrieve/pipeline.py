"""End-to-end retrieve pipeline composing everything from Phases 2-3.

  expand(query) -> sub-queries
  for each sub-query: hybrid (dense+sparse, RRF server-side)
  cross-query RRF merge
  1-hop wikilink graph expansion
  cross-encoder rerank -> top_k

Designed so the LangGraph agent (Phase 5) just calls :func:`retrieve` once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from akb.config import RetrieveConfig, load_settings
from akb.ingest.graph import VaultGraph
from akb.retrieve.graph_expand import expand as graph_expand
from akb.retrieve.hybrid import HybridRetriever, RetrievalRequest
from akb.retrieve.rerank import rerank
from akb.schemas import RetrievedChunk


@dataclass
class PipelineResult:
    query: str
    sub_queries: list[str]
    chunks: list[RetrievedChunk]
    used_reranker: bool
    used_graph_expand: bool


def retrieve(
    query: str,
    *,
    n_results: int | None = None,
    top_k: int | None = None,
    use_hyde: bool = False,
    use_decomp: bool = True,
    filter: dict[str, Any] | None = None,
    graph: VaultGraph | None = None,
    cfg: RetrieveConfig | None = None,
) -> PipelineResult:
    """High-level retrieval entry. Composes everything from Phases 2-3."""
    cfg = cfg or load_settings().retrieve
    hybrid = HybridRetriever(cfg=cfg)
    req = RetrievalRequest(
        query=query,
        n_results=n_results,
        top_k=top_k,
        use_hyde=use_hyde,
        use_decomp=use_decomp,
        filter=filter,
    )
    candidates = hybrid.retrieve(req)
    candidates = graph_expand(candidates, graph, cfg=cfg)
    if cfg.use_colbert_rerank:
        from akb.retrieve.colbert_rerank import rerank_colbert

        colbert = rerank_colbert(query, candidates, top_k=top_k or cfg.top_k)
        if colbert is not None:
            candidates = colbert
        elif cfg.use_reranker:
            candidates = rerank(query, candidates, top_k=top_k, cfg=cfg)
    elif cfg.use_reranker:
        candidates = rerank(query, candidates, top_k=top_k, cfg=cfg)
    return PipelineResult(
        query=query,
        sub_queries=[],  # populated by query_transform.expand if caller wants to inspect
        chunks=candidates[: (top_k or cfg.top_k)],
        used_reranker=cfg.use_reranker,
        used_graph_expand=cfg.graph_expand and graph is not None,
    )
