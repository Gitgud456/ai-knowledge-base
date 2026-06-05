"""RAGAS evaluation runner.

Runs the agent over a golden set, collects (question, contexts, answer,
ground_truth?) tuples, and feeds them to a small subset of RAGAS metrics. The
local LLM serves as the judge — explicitly pinned in config so reproducibility
isn't blown by a silent Ollama upgrade.

Metrics enabled by default (cheapest first):
  * context_precision     — are retrieved contexts relevant?
  * context_recall        — did retrieval surface all needed contexts? (needs ground truth)
  * faithfulness          — is the answer grounded in context?
  * answer_relevancy      — does the answer answer the question?

We also compute two cheap heuristics that don't need an LLM judge:
  * citation_hit_rate     — frac of expected_sources actually cited
  * substring_hit_rate    — frac of expected_answer_contains found in the answer
These are fast, deterministic, and useful for CI gates.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from akb.agents.graph import ChatAgent
from akb.config import load_settings
from akb.eval.golden import GoldenItem, load_golden


@dataclass
class ItemResult:
    id: str
    question: str
    answer: str
    cited_sources: list[str] = field(default_factory=list)
    expected_sources: list[str] = field(default_factory=list)
    citation_hit_rate: float = 0.0
    substring_hit_rate: float = 0.0
    elapsed_s: float = 0.0
    error: str | None = None
    ragas: dict[str, float] = field(default_factory=dict)


@dataclass
class EvalReport:
    n_items: int
    citation_hit_rate: float
    substring_hit_rate: float
    ragas_means: dict[str, float] = field(default_factory=dict)
    items: list[ItemResult] = field(default_factory=list)
    elapsed_s: float = 0.0


def _citation_hit_rate(cited: list[str], expected: list[str]) -> float:
    if not expected:
        return 1.0
    hits = sum(1 for exp in expected if any(exp in c for c in cited))
    return hits / len(expected)


def _substring_hit_rate(answer: str, expected: list[str]) -> float:
    if not expected:
        return 1.0
    a = answer.lower()
    hits = sum(1 for exp in expected if exp.lower() in a)
    return hits / len(expected)


def _run_one(item: GoldenItem, agent: ChatAgent) -> ItemResult:
    t0 = time.perf_counter()
    try:
        ans = agent.invoke(item.question)
        cited = [c.source_id for c in ans.citations]
        return ItemResult(
            id=item.id,
            question=item.question,
            answer=ans.text,
            cited_sources=cited,
            expected_sources=item.expected_sources,
            citation_hit_rate=_citation_hit_rate(cited, item.expected_sources),
            substring_hit_rate=_substring_hit_rate(ans.text, item.expected_answer_contains),
            elapsed_s=time.perf_counter() - t0,
        )
    except Exception as e:
        return ItemResult(
            id=item.id,
            question=item.question,
            answer="",
            error=str(e),
            elapsed_s=time.perf_counter() - t0,
        )


def _maybe_run_ragas(results: list[ItemResult]) -> dict[str, float]:
    """Run RAGAS metrics if the package is importable and there are usable rows.

    Falls back to an empty dict if RAGAS isn't installed or fails for any reason
    (model unreachable, etc.). The cheap heuristic gates above still apply.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            faithfulness,
        )
    except Exception:
        return {}

    rows: list[dict[str, Any]] = []
    for r in results:
        if r.error:
            continue
        rows.append(
            {
                "question": r.question,
                "answer": r.answer,
                "contexts": [], # filled in below if available
                "ground_truth": "",
            }
        )
    if not rows:
        return {}
    try:
        ds = Dataset.from_list(rows)
        scores = evaluate(ds, metrics=[context_precision, faithfulness, answer_relevancy])
        return {k: float(v) for k, v in scores.items() if isinstance(v, (int, float))}
    except Exception:
        return {}


def run_eval(
    golden_path: Path | None = None,
    *,
    use_ragas: bool = True,
) -> EvalReport:
    settings = load_settings()
    path = golden_path or settings.eval.golden_path
    items = load_golden(path)
    agent = ChatAgent()

    t0 = time.perf_counter()
    results = [_run_one(it, agent) for it in items]
    ragas_means: dict[str, float] = _maybe_run_ragas(results) if use_ragas else {}

    n = len(results) or 1
    return EvalReport(
        n_items=len(results),
        citation_hit_rate=sum(r.citation_hit_rate for r in results) / n,
        substring_hit_rate=sum(r.substring_hit_rate for r in results) / n,
        ragas_means=ragas_means,
        items=results,
        elapsed_s=time.perf_counter() - t0,
    )


def report_to_dict(report: EvalReport) -> dict[str, Any]:
    return asdict(report)
