"""Cross-encoder reranker.

Default model: ``BAAI/bge-reranker-v2-m3`` — multilingual, strong on long-tail
queries, a clean strict upgrade over the legacy ``ms-marco-MiniLM-L-6-v2``.

We expose two layers:
  * :class:`Reranker` — protocol so the pipeline can swap implementations
    (e.g. ColBERT in Phase 9) without touching callers.
  * :func:`rerank` — module-level helper that lazily builds + caches the singleton.

The reranker is a *pure* function over ``RetrievedChunk`` lists: it does not
re-fetch anything from Qdrant; it only re-scores and re-orders.

Recency weighting (opt-in via config) multiplies the cross-encoder score by
``exp(-age_days / half_life_days)`` so an older note has to be significantly
more semantically relevant to outrank a recent one.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from functools import lru_cache
from typing import Protocol

from akb.config import RetrieveConfig, load_settings
from akb.schemas import RetrievedChunk


class Reranker(Protocol):
    def score(self, query: str, candidates: list[str]) -> list[float]: ...


class _BgeReranker:
    """Sentence-Transformers CrossEncoder loaded with the BGE reranker weights."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model = None  # lazy

    def _load(self) -> object:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name)
        return self._model

    def score(self, query: str, candidates: list[str]) -> list[float]:
        if not candidates:
            return []
        pairs = [[query, c] for c in candidates]
        scores = self._load().predict(pairs, show_progress_bar=False)  # type: ignore[attr-defined]
        return [float(s) for s in scores]


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    cfg = load_settings().retrieve
    return _BgeReranker(cfg.reranker_model)


def _age_days(rc: RetrievedChunk, now: datetime) -> float | None:
    """Best-effort chunk age in days. Returns None if no timestamp is available."""
    ts = rc.chunk.metadata.get("modified_at") or rc.chunk.metadata.get("created_at")
    if not isinstance(ts, str):
        return None
    try:
        dt = datetime.fromisoformat(ts)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    delta = now - dt
    return max(delta.total_seconds() / 86400.0, 0.0)


def _apply_recency(
    candidates: list[RetrievedChunk],
    weight: float,
    half_life_days: float,
) -> None:
    """Mutate ``rerank_score`` in place with a recency multiplier.

    ``weight`` interpolates between "no recency effect" (0.0) and "full decay"
    (1.0). Chunks without a usable timestamp keep their score unchanged.
    """
    if weight <= 0.0 or half_life_days <= 0.0:
        return
    now = datetime.now(timezone.utc)
    decay_k = math.log(2) / half_life_days
    for rc in candidates:
        age = _age_days(rc, now)
        if age is None or rc.rerank_score is None:
            continue
        decay = math.exp(-decay_k * age)
        # weight=0 -> 1.0 (no effect); weight=1 -> full decay
        mult = (1.0 - weight) + weight * decay
        rc.rerank_score = rc.rerank_score * mult


def rerank(
    query: str,
    candidates: list[RetrievedChunk],
    *,
    top_k: int | None = None,
    cfg: RetrieveConfig | None = None,
    reranker: Reranker | None = None,
) -> list[RetrievedChunk]:
    """Re-score and re-order candidates by a cross-encoder.

    Operates only on the top ``reranker_top_n`` candidates (configurable) — the
    long tail rarely beats the head after RRF, so paying cross-encoder cost on
    all of them is wasteful.
    """
    cfg = cfg or load_settings().retrieve
    if not candidates:
        return []
    pool = candidates[: cfg.reranker_top_n]
    rest = candidates[cfg.reranker_top_n :]
    reranker = reranker or get_reranker()

    # Use contextualized_text if present (Phase 4); the model wants the same
    # text the embedder saw.
    texts = [c.chunk.embed_text for c in pool]
    scores = reranker.score(query, texts)
    for rc, s in zip(pool, scores):
        rc.rerank_score = s

    _apply_recency(pool, cfg.recency_weight, cfg.recency_half_life_days)

    pool.sort(key=lambda rc: rc.rerank_score or 0.0, reverse=True)
    final = pool + rest
    k = top_k or cfg.top_k
    return final[:k]
