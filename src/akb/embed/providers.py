"""Embedding providers.

Default: ``BGE-M3`` via FlagEmbedding, which gives us *dense + lexical-sparse* in
one pass — exactly what Qdrant's hybrid search wants. The model is heavy (~2.3GB),
so we lazy-load on first use and cache the singleton.

Sparse output is converted to Qdrant's (indices, values) shape; the dense side is
L2-normalised so cosine == dot product.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import numpy as np

from akb.config import EmbedConfig, load_settings


@dataclass(frozen=True)
class SparseVec:
    indices: list[int]
    values: list[float]


@dataclass(frozen=True)
class EmbeddedBatch:
    dense: list[list[float]]
    sparse: list[SparseVec]


class Embedder(Protocol):
    dim: int
    use_sparse: bool

    def embed_documents(self, texts: list[str]) -> EmbeddedBatch: ...
    def embed_query(self, text: str) -> EmbeddedBatch: ...


class _BgeM3Embedder:
    def __init__(self, cfg: EmbedConfig) -> None:
        self._cfg = cfg
        self._model = None  # lazy

    @property
    def dim(self) -> int:
        return self._cfg.dim

    @property
    def use_sparse(self) -> bool:
        return self._cfg.use_sparse

    def _load(self) -> object:
        if self._model is None:
            from FlagEmbedding import BGEM3FlagModel  # type: ignore[import-untyped]

            self._model = BGEM3FlagModel(self._cfg.model, use_fp16=True)
        return self._model

    def _encode(self, texts: list[str]) -> EmbeddedBatch:
        if not texts:
            return EmbeddedBatch(dense=[], sparse=[])
        model = self._load()
        result = model.encode(  # type: ignore[attr-defined]
            texts,
            batch_size=self._cfg.batch_size,
            return_dense=True,
            return_sparse=self._cfg.use_sparse,
            return_colbert_vecs=False,
        )
        dense_np: "np.ndarray" = result["dense_vecs"]
        if self._cfg.normalize:
            import numpy as np

            norms = np.linalg.norm(dense_np, axis=1, keepdims=True) + 1e-12
            dense_np = dense_np / norms
        dense = dense_np.tolist()

        sparse: list[SparseVec] = []
        if self._cfg.use_sparse:
            for weights in result["lexical_weights"]:
                idx: list[int] = []
                val: list[float] = []
                for token_id, weight in weights.items():
                    idx.append(int(token_id))
                    val.append(float(weight))
                sparse.append(SparseVec(indices=idx, values=val))
        else:
            sparse = [SparseVec(indices=[], values=[]) for _ in texts]
        return EmbeddedBatch(dense=dense, sparse=sparse)

    def embed_documents(self, texts: list[str]) -> EmbeddedBatch:
        return self._encode(texts)

    def embed_query(self, text: str) -> EmbeddedBatch:
        return self._encode([text])


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    cfg = load_settings().embed
    return _BgeM3Embedder(cfg)
