"""Speculative RAG: context partitioning, JSON parsing, and the verifier
fallback paths. Heavy LLM calls are mocked."""

from __future__ import annotations

import json
from typing import Any

import pytest

from akb.agents import speculative as sp
from akb.config import SpeculativeConfig
from akb.schemas import Citation


def test_partition_round_robin() -> None:
    out = sp._partition(["a", "b", "c", "d", "e"], n_drafts=2)
    assert out == [["a", "c", "e"], ["b", "d"]]


def test_partition_drops_empty_subsets_when_too_few_chunks() -> None:
    out = sp._partition(["only-one"], n_drafts=4)
    assert out == [["only-one"]]


def test_partition_n_drafts_one_keeps_everything() -> None:
    out = sp._partition(["a", "b", "c"], n_drafts=1)
    assert out == [["a", "b", "c"]]


def test_verify_parses_clean_json(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake(**kwargs: Any) -> dict[str, str]:
        return {
            "response": json.dumps(
                {"best": 1, "score": 8.5, "rationale": "draft 1 was more grounded"}
            )
        }

    monkeypatch.setattr(sp.ollama, "generate", _fake)
    best, score, rationale = sp._verify(["d0", "d1"], query="q", model="m")
    assert best == 1
    assert score == 8.5
    assert "grounded" in rationale


def test_verify_tolerates_extra_text(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake(**kwargs: Any) -> dict[str, str]:
        return {
            "response": "Here is my answer:\n" + json.dumps({"best": 0, "score": 7.0})
        }

    monkeypatch.setattr(sp.ollama, "generate", _fake)
    best, score, _ = sp._verify(["d0", "d1"], query="q", model="m")
    assert best == 0
    assert score == 7.0


def test_verify_returns_zero_on_garbage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sp.ollama, "generate", lambda **kw: {"response": "no json"})
    best, score, rationale = sp._verify(["d0"], query="q", model="m")
    assert best == 0
    assert score == 0.0
    assert rationale == ""


def test_run_speculative_picks_winner(monkeypatch: pytest.MonkeyPatch) -> None:
    drafts: dict[int, str] = {0: "draft zero", 1: "draft one", 2: "draft two"}
    chat_calls: list[int] = []

    def _fake_chat(**kwargs: Any) -> dict[str, dict[str, str]]:
        prompt = kwargs["messages"][0]["content"]
        # The prompt contains "CONTEXT SUBSET <i>:"
        import re

        m = re.search(r"CONTEXT SUBSET (\d+):", prompt)
        i = int(m.group(1)) if m else 0
        chat_calls.append(i)
        return {"message": {"content": drafts[i]}}

    def _fake_generate(**kwargs: Any) -> dict[str, str]:
        return {
            "response": json.dumps({"best": 2, "score": 9.0, "rationale": "winner"})
        }

    monkeypatch.setattr(sp.ollama, "chat", _fake_chat)
    monkeypatch.setattr(sp.ollama, "generate", _fake_generate)

    res = sp.run_speculative(
        query="q",
        context_chunks=["c0", "c1", "c2", "c3", "c4", "c5"],
        citations=[],
        cfg=SpeculativeConfig(enabled=True, n_drafts=3),
    )
    assert sorted(chat_calls) == [0, 1, 2]
    assert res.best_id == 2
    assert res.answer == "draft two"
    assert res.verifier_score == 9.0


def test_run_speculative_single_draft_skips_verifier(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sp.ollama, "chat", lambda **kw: {"message": {"content": "single"}}
    )
    called: list[bool] = []
    monkeypatch.setattr(
        sp.ollama,
        "generate",
        lambda **kw: called.append(True) or {"response": "{}"},
    )
    res = sp.run_speculative(
        query="q",
        context_chunks=["only"],
        citations=[],
        cfg=SpeculativeConfig(enabled=True, n_drafts=4),
    )
    assert res.answer == "single"
    assert called == []  # verifier was NOT called


def test_run_speculative_handles_empty_drafts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sp.ollama, "chat", lambda **kw: {"message": {"content": ""}})
    res = sp.run_speculative(
        query="q",
        context_chunks=["a", "b", "c"],
        citations=[],
        cfg=SpeculativeConfig(enabled=True, n_drafts=3),
    )
    assert "no drafter" in res.answer
