"""Qdrant embedded vector store.

Collection layout
-----------------
``knowledge_base`` with **named vectors**:
  * ``dense``  — cosine, 1024-d (BGE-M3)
  * ``sparse`` — sparse vector params, IDF-on (BGE-M3 lexical weights)

Payload schema (one Qdrant point per Chunk)::

    {
      "chunk_id":      str,
      "source_id":     str,
      "source_type":   str,
      "text":          str,
      "header_path":   list[str],
      "chunk_index":   int,
      "tags":          list[str],
      "wikilinks":     list[str],
      "title":         str | None,
      "aliases":       list[str],
      "metadata":      dict[str, Any],
    }

Indexed payload fields (cheap filters): ``source_id``, ``source_type``, ``tags``,
``wikilinks``. Add more later if a query pattern justifies it.

Retrieval uses ``query_points`` with native ``Fusion.RRF`` over a dense and a
sparse prefetch — keeps fusion server-side and avoids client-side reshuffling.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models

from akb.config import EmbedConfig, load_settings
from akb.embed.providers import EmbeddedBatch, SparseVec, get_embedder
from akb.schemas import Chunk, RetrievedChunk

COLLECTION = "knowledge_base"
DENSE_NAME = "dense"
SPARSE_NAME = "sparse"


def _client(qdrant_dir: Path) -> QdrantClient:
    qdrant_dir.mkdir(parents=True, exist_ok=True)
    # Embedded mode: no daemon, on-disk persistence.
    return QdrantClient(path=str(qdrant_dir))


def _vectors_config(
    dim: int, binary_quant: bool = False
) -> dict[str, models.VectorParams]:
    quantization = None
    if binary_quant:
        try:
            quantization = models.BinaryQuantization(
                binary=models.BinaryQuantizationConfig(always_ram=True)
            )
        except Exception:
            # Older qdrant-client: skip silently; we'll warn at collection-create time.
            quantization = None
    return {
        DENSE_NAME: models.VectorParams(
            size=dim,
            distance=models.Distance.COSINE,
            quantization_config=quantization,
        ),
    }


def _sparse_vectors_config() -> dict[str, models.SparseVectorParams]:
    return {
        SPARSE_NAME: models.SparseVectorParams(
            index=models.SparseIndexParams(on_disk=False),
            modifier=models.Modifier.IDF,
        ),
    }


def _point_id(chunk_id: str) -> str:
    # Qdrant accepts UUIDs and unsigned ints. uuid5 over a deterministic chunk_id
    # (see schemas.Chunk._ensure_deterministic_id) gives us idempotent upserts —
    # re-running `akb ingest` on an unchanged file does not duplicate points.
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def _payload(chunk: Chunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "source_id": chunk.source_id,
        "source_type": chunk.source_type.value,
        "text": chunk.text,
        "contextualized_text": chunk.contextualized_text,
        "header_path": chunk.header_path,
        "chunk_index": chunk.chunk_index,
        "tags": chunk.tags,
        "wikilinks": chunk.wikilinks,
        "title": chunk.metadata.get("title"),
        "aliases": chunk.metadata.get("aliases", []),
        "frontmatter_keys": chunk.metadata.get("frontmatter_keys", []),
        "created_at": chunk.metadata.get("created_at"),
        "modified_at": chunk.metadata.get("modified_at"),
    }


def _hydrate(payload: dict[str, Any]) -> Chunk:
    from akb.schemas import SourceType

    return Chunk(
        chunk_id=payload["chunk_id"],
        source_id=payload["source_id"],
        source_type=SourceType(payload["source_type"]),
        text=payload["text"],
        contextualized_text=payload.get("contextualized_text"),
        header_path=payload.get("header_path", []),
        chunk_index=payload.get("chunk_index", 0),
        tags=payload.get("tags", []),
        wikilinks=payload.get("wikilinks", []),
        metadata={
            "title": payload.get("title"),
            "aliases": payload.get("aliases", []),
            "frontmatter_keys": payload.get("frontmatter_keys", []),
            "created_at": payload.get("created_at"),
            "modified_at": payload.get("modified_at"),
        },
    )


def _is_local_client(client: QdrantClient) -> bool:
    try:
        from qdrant_client.local.qdrant_local import QdrantLocal

        return isinstance(getattr(client, "_client", None), QdrantLocal)
    except Exception:
        return False


def _ensure_payload_indices(client: QdrantClient) -> None:
    # Local/embedded Qdrant ignores payload indices and warns loudly each time.
    # On a real server they speed up keyword filters significantly.
    if _is_local_client(client):
        return
    for field, schema in (
        ("source_id", models.PayloadSchemaType.KEYWORD),
        ("source_type", models.PayloadSchemaType.KEYWORD),
        ("tags", models.PayloadSchemaType.KEYWORD),
        ("wikilinks", models.PayloadSchemaType.KEYWORD),
    ):
        try:
            client.create_payload_index(COLLECTION, field_name=field, field_schema=schema)
        except Exception:
            # already exists — fine
            pass


class QdrantStore:
    """Thin wrapper around the embedded Qdrant client.

    Owns the collection schema; everyone else goes through this.
    """

    def __init__(self, qdrant_dir: Path | None = None, embed_cfg: EmbedConfig | None = None) -> None:
        settings = load_settings()
        self._dir = qdrant_dir or settings.paths.qdrant_dir
        self._embed_cfg = embed_cfg or settings.embed
        self._client = _client(self._dir)
        self._ensure_collection()
        _ensure_payload_indices(self._client)

    @property
    def client(self) -> QdrantClient:
        return self._client

    def _ensure_collection(self) -> None:
        if self._client.collection_exists(COLLECTION):
            self._check_index_stamp()
            return
        kwargs: dict[str, Any] = {
            "collection_name": COLLECTION,
            "vectors_config": _vectors_config(
                self._embed_cfg.dim,
                binary_quant=self._embed_cfg.binary_quantization,
            ),
        }
        if self._embed_cfg.use_sparse:
            kwargs["sparse_vectors_config"] = _sparse_vectors_config()
        self._client.create_collection(**kwargs)
        self._check_index_stamp()

    def _check_index_stamp(self) -> None:
        """Verify the live index was built with the same embed model + dim + quant.

        On first run, writes the stamp through. On a mismatch, logs an error
        and raises :class:`IndexCompatibilityError` — the caller should run
        ``akb reindex`` (or revert the config).
        """
        from akb.store.migrations import check_index_compatible

        res = check_index_compatible(self._client, self._embed_cfg)
        if not res.compatible:
            from akb.obs.logging import get_logger

            _log = get_logger(__name__)
            _log.error("index.stamp.mismatch", reason=res.reason)
            raise IndexCompatibilityError(res.reason)

    def recreate(self) -> None:
        """Drop and re-create the collection. Destructive.

        Also wipes the index stamp so the next start writes a fresh one for
        the new embed model / dim / quant.
        """
        from akb.store.migrations import INDEX_META_COLLECTION

        if self._client.collection_exists(COLLECTION):
            self._client.delete_collection(COLLECTION)
        if self._client.collection_exists(INDEX_META_COLLECTION):
            self._client.delete_collection(INDEX_META_COLLECTION)
        self._ensure_collection()
        _ensure_payload_indices(self._client)

    def upsert(self, chunks: list[Chunk], embeddings: EmbeddedBatch) -> int:
        if not chunks:
            return 0
        assert len(chunks) == len(embeddings.dense), "embedding/chunk length mismatch"
        points: list[models.PointStruct] = []
        for i, c in enumerate(chunks):
            vector: dict[str, Any] = {DENSE_NAME: embeddings.dense[i]}
            if self._embed_cfg.use_sparse and embeddings.sparse:
                sv = embeddings.sparse[i]
                vector[SPARSE_NAME] = models.SparseVector(indices=sv.indices, values=sv.values)
            points.append(
                models.PointStruct(
                    id=_point_id(c.chunk_id),
                    vector=vector,
                    payload=_payload(c),
                )
            )
        self._client.upsert(collection_name=COLLECTION, points=points, wait=True)
        return len(points)

    def delete_by_source(self, source_id: str) -> int:
        flt = models.Filter(
            must=[models.FieldCondition(key="source_id", match=models.MatchValue(value=source_id))]
        )
        # count first for the caller's bookkeeping
        n = self._client.count(COLLECTION, count_filter=flt, exact=True).count
        self._client.delete(
            collection_name=COLLECTION,
            points_selector=models.FilterSelector(filter=flt),
            wait=True,
        )
        return n

    def list_sources(self) -> list[str]:
        # Scroll all points but only grab source_id from payload — small per row.
        out: set[str] = set()
        offset: Any = None
        while True:
            points, offset = self._client.scroll(
                collection_name=COLLECTION,
                with_payload=["source_id"],
                with_vectors=False,
                limit=1024,
                offset=offset,
            )
            for p in points:
                sid = (p.payload or {}).get("source_id")
                if isinstance(sid, str):
                    out.add(sid)
            if offset is None:
                break
        return sorted(out)

    def count(self) -> int:
        return int(self._client.count(COLLECTION, exact=True).count)

    def search_hybrid(
        self,
        dense_vec: list[float],
        sparse_vec: SparseVec | None,
        n_results: int,
        where: models.Filter | None = None,
        rrf_k: int = 60,
    ) -> list[RetrievedChunk]:
        search_params = None
        if self._embed_cfg.binary_quantization:
            try:
                search_params = models.SearchParams(
                    quantization=models.QuantizationSearchParams(
                        ignore=False,
                        rescore=True,
                        oversampling=self._embed_cfg.binary_oversampling,
                    )
                )
            except Exception:
                search_params = None

        prefetch: list[models.Prefetch] = [
            models.Prefetch(
                query=dense_vec,
                using=DENSE_NAME,
                limit=n_results,
                filter=where,
                params=search_params,
            )
        ]
        if sparse_vec and self._embed_cfg.use_sparse and sparse_vec.indices:
            prefetch.append(
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_vec.indices, values=sparse_vec.values
                    ),
                    using=SPARSE_NAME,
                    limit=n_results,
                    filter=where,
                )
            )

        fusion_query = _build_fusion_query(rrf_k)
        res = self._client.query_points(
            collection_name=COLLECTION,
            prefetch=prefetch,
            query=fusion_query,
            limit=n_results,
            with_payload=True,
        )

        out: list[RetrievedChunk] = []
        for sp in res.points:
            chunk = _hydrate(sp.payload or {})
            out.append(RetrievedChunk(chunk=chunk, rrf_score=float(sp.score)))
        return out

    def fetch_chunks_for_sources(
        self, source_ids: Iterable[str], limit_per_source: int | None = None
    ) -> list[Chunk]:
        out: list[Chunk] = []
        for sid in source_ids:
            flt = models.Filter(
                must=[models.FieldCondition(key="source_id", match=models.MatchValue(value=sid))]
            )
            offset: Any = None
            collected = 0
            while True:
                points, offset = self._client.scroll(
                    collection_name=COLLECTION,
                    scroll_filter=flt,
                    with_payload=True,
                    with_vectors=False,
                    limit=128,
                    offset=offset,
                )
                for p in points:
                    out.append(_hydrate(p.payload or {}))
                    collected += 1
                    if limit_per_source and collected >= limit_per_source:
                        offset = None
                        break
                if offset is None:
                    break
        return out


class IndexCompatibilityError(RuntimeError):
    """Raised when the live Qdrant index was built with a different embed
    model, dim, or quantization setting than the active config. Run
    ``akb reindex`` (or revert the config) to recover."""


_SINGLETON: QdrantStore | None = None
_SINGLETON_LOCK = threading.Lock()


def _build_fusion_query(rrf_k: int) -> Any:
    """Build a FusionQuery, passing ``params`` if the installed qdrant-client supports it.

    Older clients (<1.12) don't expose ``FusionParams``; in that case fall back to the
    default ``k`` (which is also 60, matching our config default — but a non-60 value
    in config would silently no-op on old clients).
    """
    params_cls = getattr(models, "FusionParams", None)
    if params_cls is not None:
        try:
            return models.FusionQuery(fusion=models.Fusion.RRF, params=params_cls(k=rrf_k))
        except Exception:
            pass
    return models.FusionQuery(fusion=models.Fusion.RRF)


def get_store() -> QdrantStore:
    """Process-level singleton. Thread-safe: embedded Qdrant locks its data dir,
    so two concurrent inits (Streamlit + ``akb sync --watch`` sharing a process)
    would crash on the second. Lock guarantees one init."""
    global _SINGLETON
    if _SINGLETON is None:
        with _SINGLETON_LOCK:
            if _SINGLETON is None:
                _SINGLETON = QdrantStore()
    return _SINGLETON


def reset_store_singleton() -> None:
    """Drop the cached store — useful in tests."""
    global _SINGLETON
    with _SINGLETON_LOCK:
        _SINGLETON = None


__all__ = [
    "QdrantStore",
    "get_store",
    "COLLECTION",
    "DENSE_NAME",
    "SPARSE_NAME",
    "get_embedder",
]
