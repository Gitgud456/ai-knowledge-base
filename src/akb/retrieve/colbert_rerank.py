"""Optional ColBERT (late-interaction) reranker via RAGatouille.

Trade-off vs cross-encoder rerank:
  + Better recall on long-tail queries
  + Token-level matching (great on proper nouns, code)
  -  Much bigger index (multi-vector); slower to load
  -  Requires the ``[late]`` optional dependency group

Loaded lazily. If ``ragatouille`` isn't installed, callers should fall back to
:mod:`akb.retrieve.rerank`. The wiring in :mod:`akb.retrieve.pipeline` checks a
config flag and falls back automatically.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Protocol

from akb.schemas import RetrievedChunk


class ColBERTReranker(Protocol):
    def score(self, query: str, passages: list[str]) -> list[float]: ...


class _RagatouilleReranker:
    def __init__(self, model_name: str = "colbert-ir/colbertv2.0") -> None:
        self._model_name = model_name
        self._model = None

    def _load(self) -> object:
        if self._model is None:
            from ragatouille import RAGPretrainedModel  # type: ignore[import-untyped]

            self._model = RAGPretrainedModel.from_pretrained(self._model_name)
        return self._model

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        model = self._load()
        out = model.rerank(query=query, documents=passages, k=len(passages))  # type: ignore[attr-defined]
        # ragatouille returns rows ordered by score; rebuild original order
        score_by_idx = {row["result_index"]: float(row["score"]) for row in out}
        return [score_by_idx.get(i, 0.0) for i in range(len(passages))]


@lru_cache(maxsize=1)
def get_colbert_reranker() -> ColBERTReranker | None:
    try:
        return _RagatouilleReranker()
    except Exception:
        return None


def rerank_colbert(
    query: str,
    candidates: list[RetrievedChunk],
    top_k: int,
) -> list[RetrievedChunk] | None:
    r = get_colbert_reranker()
    if r is None or not candidates:
        return None
    scores = r.score(query, [c.chunk.embed_text for c in candidates])
    for rc, s in zip(candidates, scores):
        rc.rerank_score = s
    candidates.sort(key=lambda rc: rc.rerank_score or 0.0, reverse=True)
    return candidates[:top_k]
