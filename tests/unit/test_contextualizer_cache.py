"""Contextualizer cache: hit/miss accounting + autocommit per row."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from akb.config import IngestConfig, LLMConfig
from akb.ingest import contextualizer as cx
from akb.ingest.contextualizer import Contextualizer
from akb.schemas import Chunk, Document, SourceType


@pytest.fixture
def fake_doc() -> Document:
    return Document(
        source_id="obsidian:a.md",
        source_type=SourceType.obsidian,
        content="some longer document body",
        title="A",
    )


def _chunk(text: str, idx: int = 0) -> Chunk:
    return Chunk(source_id="obsidian:a.md", source_type=SourceType.obsidian, text=text, chunk_index=idx)


def test_contextualizer_cache_miss_then_hit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_doc: Document,
) -> None:
    calls: list[str] = []

    def _fake_ollama_generate(**kwargs: Any) -> dict[str, str]:
        calls.append(kwargs["prompt"][:20])
        return {"response": "CTX"}

    monkeypatch.setattr(cx.ollama, "generate", _fake_ollama_generate)

    c = Contextualizer(
        cache_path=tmp_path / "ctx.db",
        llm_cfg=LLMConfig(context_model="llama3:8b-instruct-q4_K_M"),
        ingest_cfg=IngestConfig(contextual_retrieval=True),
    )

    chunks = [_chunk("alpha", 0), _chunk("beta", 1)]
    c.contextualize(fake_doc, chunks)
    assert all(ch.contextualized_text and "CTX" in ch.contextualized_text for ch in chunks)
    first_calls = len(calls)
    assert first_calls == 2

    # Second pass: pure cache hit, no new LLM calls
    chunks2 = [_chunk("alpha", 0), _chunk("beta", 1)]
    c.contextualize(fake_doc, chunks2)
    assert len(calls) == first_calls
    assert all(ch.contextualized_text == "CTX\n\n" + ch.text for ch in chunks2)


def test_contextualizer_survives_mid_doc_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_doc: Document,
) -> None:
    """First pass: process chunk 0 OK, then raise on chunk 1. Second pass:
    chunk 0 should be a cache hit (the autocommit-per-row contract)."""
    state = {"call": 0}

    def _flaky(**kwargs: Any) -> dict[str, str]:
        state["call"] += 1
        if state["call"] == 2:
            raise RuntimeError("simulated interrupt")
        return {"response": "CTX"}

    monkeypatch.setattr(cx.ollama, "generate", _flaky)

    c = Contextualizer(
        cache_path=tmp_path / "ctx.db",
        llm_cfg=LLMConfig(context_model="x"),
        ingest_cfg=IngestConfig(contextual_retrieval=True),
    )

    chunks = [_chunk("alpha", 0), _chunk("beta", 1)]
    c.contextualize(fake_doc, chunks)
    # chunk 0 got CTX; chunk 1 generation raised, _generate caught it and returned ""
    assert chunks[0].contextualized_text and chunks[0].contextualized_text.startswith("CTX")

    # New contextualizer (simulates next process); chunk 0 must come from cache
    state["call"] = 0  # reset counter

    def _ok(**kwargs: Any) -> dict[str, str]:
        state["call"] += 1
        return {"response": "CTX2"}

    monkeypatch.setattr(cx.ollama, "generate", _ok)
    c2 = Contextualizer(
        cache_path=tmp_path / "ctx.db",
        llm_cfg=LLMConfig(context_model="x"),
        ingest_cfg=IngestConfig(contextual_retrieval=True),
    )
    chunks2 = [_chunk("alpha", 0), _chunk("beta", 1)]
    c2.contextualize(fake_doc, chunks2)
    # chunk 0 hit cache -> contains "CTX" not "CTX2"; chunk 1 was a fresh miss
    assert chunks2[0].contextualized_text and "CTX" in chunks2[0].contextualized_text
    assert chunks2[0].contextualized_text and "CTX2" not in chunks2[0].contextualized_text
    assert chunks2[1].contextualized_text and "CTX2" in chunks2[1].contextualized_text
