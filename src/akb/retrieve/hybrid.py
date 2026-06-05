"""High-level hybrid retriever.

  expand(query) -> [sub-queries]                       (Ollama)
  for each sub-query:
      embed -> dense + sparse vectors                  (BGE-M3)
      qdrant.search_hybrid(prefetch=[dense, sparse], fusion=RRF) -> top n_results
  RRF-fuse across sub-queries                          (client side, simple RRF)
  return top n_results, ready for the reranker (Phase 3)
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import time

from akb.config import RetrieveConfig, load_settings
from akb.embed.providers import get_embedder
from akb.obs.logging import get_logger
from akb.retrieve.query_transform import expand
from akb.schemas import RetrievedChunk
from akb.store.qdrant_store import QdrantStore, get_store

log = get_logger(__name__)

try:  # Qdrant filter models are only available with the package installed.
    from qdrant_client import models  # type: ignore[import-untyped]
except Exception:  # pragma: no cover
    models = None  # type: ignore[assignment]


@dataclass(frozen=True)
class RetrievalRequest:
    query: str
    n_results: int | None = None
    top_k: int | None = None
    use_hyde: bool = False
    use_decomp: bool = True
    filter: dict[str, Any] | None = None


def _client_filter(filter_payload: dict[str, Any] | None) -> Any | None:
    """Translate a small dict-style filter into a Qdrant Filter.

    Supports:
        {"source_id": "..."}                                    exact match
        {"source_type": "obsidian"}                             exact match
        {"tags": ["security", "ai"]}                            any-of
        {"wikilinks": ["Some Note"]}                            any-of
        {"modified_at": {"gte": "...", "lte": "..."}}           range
    """
    if not filter_payload or models is None:
        return None
    must: list[Any] = []
    for key, value in filter_payload.items():
        if isinstance(value, list):
            must.append(
                models.FieldCondition(key=key, match=models.MatchAny(any=value))
            )
        elif isinstance(value, dict) and ("gte" in value or "lte" in value or "gt" in value or "lt" in value):
            must.append(
                models.FieldCondition(
                    key=key,
                    range=models.DatetimeRange(
                        gte=value.get("gte"),
                        gt=value.get("gt"),
                        lte=value.get("lte"),
                        lt=value.get("lt"),
                    ),
                )
            )
        else:
            must.append(
                models.FieldCondition(key=key, match=models.MatchValue(value=value))
            )
    return models.Filter(must=must) if must else None


def _rrf_merge(
    bucket: Iterable[list[RetrievedChunk]],
    k: int,
) -> list[RetrievedChunk]:
    """Aggregate multiple ranked lists into one using Reciprocal Rank Fusion.

    Server-side RRF already handles the dense vs sparse fusion *within* a single
    sub-query. This RRF fuses *across* sub-queries (multi-query / HyDE).
    """
    scores: dict[str, float] = {}
    best: dict[str, RetrievedChunk] = {}
    for ranked in bucket:
        for rank, hit in enumerate(ranked):
            key = hit.chunk.chunk_id
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            prev = best.get(key)
            if prev is None or (hit.rrf_score or 0) > (prev.rrf_score or 0):
                best[key] = hit
    fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    out: list[RetrievedChunk] = []
    for cid, s in fused:
        rc = best[cid]
        rc.rrf_score = s
        out.append(rc)
    return out


class HybridRetriever:
    def __init__(
        self,
        store: QdrantStore | None = None,
        cfg: RetrieveConfig | None = None,
    ) -> None:
        self._store = store or get_store()
        self._cfg = cfg or load_settings().retrieve
        self._embedder = get_embedder()

    def _resolved(self, req: RetrievalRequest) -> tuple[int, int]:
        n_results = req.n_results or self._cfg.n_results
        top_k = req.top_k or self._cfg.top_k
        return n_results, top_k

    def retrieve(self, req: RetrievalRequest) -> list[RetrievedChunk]:
        n_results, top_k = self._resolved(req)
        t0 = time.perf_counter()
        sub_queries = expand(req.query, use_hyde=req.use_hyde, use_decomp=req.use_decomp)
        flt = _client_filter(req.filter)

        per_query: list[list[RetrievedChunk]] = []
        for sq in sub_queries:
            emb = self._embedder.embed_query(sq)
            dense = emb.dense[0]
            sparse = emb.sparse[0] if emb.sparse else None
            hits = self._store.search_hybrid(
                dense_vec=dense,
                sparse_vec=sparse,
                n_results=n_results,
                where=flt,
                rrf_k=self._cfg.rrf_k,
            )
            per_query.append(hits)

        merged = _rrf_merge(per_query, k=self._cfg.rrf_k)
        log.info(
            "retrieve.hybrid",
            sub_queries=len(sub_queries),
            candidates=len(merged),
            elapsed_ms=int((time.perf_counter() - t0) * 1000),
            has_filter=bool(flt),
        )
        return merged[:max(n_results, top_k)]
