"""End-to-end integration test for the retrieve pipeline.

Uses:
  * an in-memory Qdrant (no disk, no daemon),
  * a deterministic fake embedder (bag-of-words → dense + sparse),
  * the real chunker, dedupe, hybrid retriever, RRF merge, and rerank wiring
    (rerank is stubbed because we don't want to download bge-reranker here).

Proves the pipeline wires correctly: chunks → upsert → hybrid → graph_expand → rerank → top_k.
"""

from __future__ import annotations

import hashlib

import pytest
from qdrant_client import QdrantClient

from akb.embed.providers import EmbeddedBatch, SparseVec
from akb.retrieve.hybrid import HybridRetriever, RetrievalRequest
from akb.retrieve.pipeline import retrieve as retrieve_top
from akb.schemas import Chunk, SourceType
from akb.store import qdrant_store as qs


class _FakeEmbedder:
    """Bag-of-words → 16-d dense + token-hash sparse. Deterministic, fast."""

    dim = 16
    use_sparse = True

    def _vectors(self, texts: list[str]) -> EmbeddedBatch:
        dense: list[list[float]] = []
        sparse: list[SparseVec] = []
        for text in texts:
            tokens = [t.lower() for t in text.split() if t]
            vec = [0.0] * self.dim
            for tok in tokens:
                bucket = int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim
                vec[bucket] += 1.0
            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            dense.append([v / norm for v in vec])
            indices: list[int] = []
            values: list[float] = []
            for tok in set(tokens):
                idx = int(hashlib.md5(tok.encode()).hexdigest(), 16) % 100000
                indices.append(idx)
                values.append(1.0)
            sparse.append(SparseVec(indices=indices, values=values))
        return EmbeddedBatch(dense=dense, sparse=sparse)

    def embed_documents(self, texts: list[str]) -> EmbeddedBatch:
        return self._vectors(texts)

    def embed_query(self, text: str) -> EmbeddedBatch:
        return self._vectors([text])


@pytest.fixture
def fake_embedder() -> _FakeEmbedder:
    return _FakeEmbedder()


@pytest.fixture
def in_memory_store(monkeypatch: pytest.MonkeyPatch, fake_embedder: _FakeEmbedder) -> qs.QdrantStore:
    """Wire a Qdrant in-memory client + fake embedder + low-dim collection."""
    from akb.config import EmbedConfig

    # Patch at every site that imported `get_embedder` into its own namespace.
    monkeypatch.setattr("akb.embed.providers.get_embedder", lambda: fake_embedder)
    monkeypatch.setattr("akb.store.qdrant_store.get_embedder", lambda: fake_embedder)
    monkeypatch.setattr("akb.retrieve.hybrid.get_embedder", lambda: fake_embedder)
    monkeypatch.setattr("akb.ingest.upsert.get_embedder", lambda: fake_embedder)

    cfg = EmbedConfig(model="fake", dim=fake_embedder.dim, use_sparse=True, normalize=False, batch_size=8)

    class _MemStore(qs.QdrantStore):
        def __init__(self) -> None:
            self._dir = None  # type: ignore[assignment]
            self._embed_cfg = cfg
            self._client = QdrantClient(":memory:")
            self._ensure_collection()
            qs._ensure_payload_indices(self._client)

    store = _MemStore()
    monkeypatch.setattr("akb.store.qdrant_store.get_store", lambda: store)
    monkeypatch.setattr("akb.retrieve.hybrid.get_store", lambda: store)
    monkeypatch.setattr("akb.retrieve.graph_expand.get_store", lambda: store)
    return store


def _chunk(text: str, source_id: str, idx: int = 0) -> Chunk:
    return Chunk(
        source_id=source_id,
        source_type=SourceType.obsidian,
        text=text,
        chunk_index=idx,
        metadata={"title": source_id.split(":")[-1]},
    )


def _upsert(store: qs.QdrantStore, embedder: _FakeEmbedder, chunks: list[Chunk]) -> int:
    return store.upsert(chunks, embedder.embed_documents([c.text for c in chunks]))


def test_round_trip_hybrid_retrieval(
    fake_embedder: _FakeEmbedder,
    in_memory_store: qs.QdrantStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = [
        _chunk("ARP spoofing intercepts traffic on a local network", "obsidian:arp.md", 0),
        _chunk("Python decorators wrap functions for side effects", "obsidian:py.md", 0),
        _chunk("Network protocols include TCP UDP and IP", "obsidian:net.md", 0),
        _chunk("Cooking pasta requires boiling water and salt", "obsidian:pasta.md", 0),
    ]
    n = _upsert(in_memory_store, fake_embedder, chunks)
    assert n == 4
    assert in_memory_store.count() == 4

    # Disable LLM-driven query transforms so we just test the retrieval wiring.
    monkeypatch.setattr("akb.retrieve.hybrid.expand", lambda q, **kw: [q])

    hr = HybridRetriever(store=in_memory_store)
    hits = hr.retrieve(RetrievalRequest(query="arp network spoof", n_results=4, top_k=4))
    sources = [h.chunk.source_id for h in hits]
    assert "obsidian:arp.md" in sources, f"expected arp.md in top hits, got {sources}"
    assert "obsidian:pasta.md" not in sources[:2]


def test_pipeline_with_filter_and_no_rerank(
    fake_embedder: _FakeEmbedder,
    in_memory_store: qs.QdrantStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = [
        _chunk("ARP spoofing description here", "obsidian:arp.md", 0),
        _chunk("ARP cache poisoning details", "obsidian:arp.md", 1),
        _chunk("Python decorators chunk", "obsidian:py.md", 0),
    ]
    chunks[0].tags = ["security"]
    chunks[1].tags = ["security"]
    chunks[2].tags = ["python"]
    _upsert(in_memory_store, fake_embedder, chunks)

    monkeypatch.setattr("akb.retrieve.hybrid.expand", lambda q, **kw: [q])
    # Turn off rerank + graph expand for this test
    from akb.config import RetrieveConfig

    cfg = RetrieveConfig(use_reranker=False, graph_expand=False, n_results=10, top_k=5)

    res = retrieve_top(
        "ARP",
        filter={"tags": ["security"]},
        cfg=cfg,
    )
    for c in res.chunks:
        assert "security" in c.chunk.tags
    assert all(c.chunk.source_id == "obsidian:arp.md" for c in res.chunks)


def test_delete_by_source(
    fake_embedder: _FakeEmbedder,
    in_memory_store: qs.QdrantStore,
) -> None:
    chunks = [
        _chunk("first chunk", "obsidian:a.md", 0),
        _chunk("second chunk", "obsidian:a.md", 1),
        _chunk("other source", "obsidian:b.md", 0),
    ]
    _upsert(in_memory_store, fake_embedder, chunks)
    assert in_memory_store.count() == 3
    n = in_memory_store.delete_by_source("obsidian:a.md")
    assert n == 2
    assert in_memory_store.count() == 1
    assert in_memory_store.list_sources() == ["obsidian:b.md"]
