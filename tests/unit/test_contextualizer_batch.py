"""Batched contextualizer: JSON-array path + per-chunk fallback on malformed JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from akb.config import IngestConfig, LLMConfig
from akb.ingest import contextualizer as cx
from akb.ingest.contextualizer import Contextualizer, _parse_batch_response
from akb.schemas import Chunk, Document, SourceType


@pytest.fixture
def doc() -> Document:
    return Document(
        source_id="obsidian:a.md",
        source_type=SourceType.obsidian,
        content="this is the document body",
        title="A",
    )


def _ch(text: str, i: int) -> Chunk:
    return Chunk(source_id="obsidian:a.md", source_type=SourceType.obsidian, text=text, chunk_index=i)


def test_parse_batch_response_well_formed() -> None:
    raw = json.dumps({"contexts": [{"id": 0, "context": "c0"}, {"id": 2, "context": "c2"}]})
    out = _parse_batch_response(raw)
    assert out == {0: "c0", 2: "c2"}


def test_parse_batch_response_tolerates_extra_text() -> None:
    raw = "Sure: " + json.dumps({"contexts": [{"id": 1, "context": "c1"}]}) + " done"
    assert _parse_batch_response(raw) == {1: "c1"}


def test_parse_batch_response_garbage_returns_empty() -> None:
    assert _parse_batch_response("not json at all") == {}


def test_batched_contextualizer_one_call_per_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    doc: Document,
) -> None:
    calls: list[int] = []

    def _fake(**kwargs: Any) -> dict[str, str]:
        # decide how many chunks this batch should answer for
        prompt = kwargs["prompt"]
        ids = [int(s) for s in __import__("re").findall(r'<chunk id="(\d+)"', prompt)]
        calls.append(len(ids))
        return {
            "response": json.dumps(
                {"contexts": [{"id": i, "context": f"ctx{i}"} for i in ids]}
            )
        }

    monkeypatch.setattr(cx.ollama, "generate", _fake)

    c = Contextualizer(
        cache_path=tmp_path / "ctx.db",
        llm_cfg=LLMConfig(context_model="x"),
        ingest_cfg=IngestConfig(contextual_retrieval=True, context_batch_size=4),
    )
    chunks = [_ch(f"chunk-{i}", i) for i in range(10)]
    c.contextualize(doc, chunks)
    # 10 chunks / batch_size 4 -> 3 batches (4, 4, 2)
    assert calls == [4, 4, 2]
    for i, ch in enumerate(chunks):
        assert ch.contextualized_text == f"ctx{i}\n\nchunk-{i}"


def test_batch_falls_back_to_single_on_malformed_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    doc: Document,
) -> None:
    flips = {"batch_called": False, "singles": 0}

    def _fake(**kwargs: Any) -> dict[str, str]:
        prompt = kwargs["prompt"]
        if "Respond with ONLY a JSON object" in prompt:
            flips["batch_called"] = True
            return {"response": "this is not json"}
        # single-chunk fallback path
        flips["singles"] += 1
        return {"response": "single-ctx"}

    monkeypatch.setattr(cx.ollama, "generate", _fake)

    c = Contextualizer(
        cache_path=tmp_path / "ctx.db",
        llm_cfg=LLMConfig(context_model="x"),
        ingest_cfg=IngestConfig(contextual_retrieval=True, context_batch_size=3),
    )
    chunks = [_ch(f"c{i}", i) for i in range(3)]
    c.contextualize(doc, chunks)
    assert flips["batch_called"]
    assert flips["singles"] == 3
    for ch in chunks:
        assert ch.contextualized_text == f"single-ctx\n\n{ch.text}"
