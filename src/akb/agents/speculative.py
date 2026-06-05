"""Speculative RAG (Wang et al., 2025).

Idea: instead of one big draft from all top-k chunks, partition the chunks
into N disjoint subsets, draft N candidate answers in **parallel**, then a
larger "verifier" model scores them and the highest-scoring candidate wins.

In practice you get
  * ~2× latency reduction (the draft step is parallel, not serial across chunks)
  * a measurable quality bump (the verifier catches drafts that aren't grounded)

This file exposes a *function* (not a LangGraph node) so the CRAG router can
opt into speculative drafting per request via ``state['speculative'] = True``.

We do NOT need a separate model for the drafter — using the same model with
shuffled context subsets gives most of the benefit. The verifier model can be
the same model with a different prompt.
"""

from __future__ import annotations

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import ollama

from akb.config import SpeculativeConfig, load_settings
from akb.obs.logging import get_logger
from akb.schemas import Citation

log = get_logger(__name__)


DRAFT_PROMPT = """You are answering from a personal knowledge base, but you only have a FRACTION of the context (one of several disjoint subsets). Answer using ONLY this subset; if it's insufficient, say so explicitly.

CONTEXT SUBSET {draft_id}:
{context}

QUESTION:
{query}

ANSWER:"""


VERIFY_PROMPT = """You are a verifier comparing several draft answers to a question. Each draft saw a different subset of the available context.

Score each draft 0-10 on:
  * Faithfulness: every claim supported by the cited context
  * Completeness: addresses the actual question
  * Conciseness: no padding

Respond with ONLY a JSON object: {{"best": <draft_id>, "score": <0-10>, "rationale": "..."}}.

QUESTION: {query}

DRAFTS:
{drafts}
"""


@dataclass
class SpeculativeResult:
    answer: str
    best_id: int
    verifier_score: float
    rationale: str
    drafts: list[str]


def _partition(context_chunks: list[str], n_drafts: int) -> list[list[str]]:
    """Deal chunks round-robin into N subsets. Keeps each subset balanced."""
    if n_drafts < 1:
        n_drafts = 1
    out: list[list[str]] = [[] for _ in range(n_drafts)]
    for i, ch in enumerate(context_chunks):
        out[i % n_drafts].append(ch)
    # Drop empty subsets (rare: fewer chunks than drafts requested)
    return [s for s in out if s]


def _draft_one(draft_id: int, ctx: str, query: str, model: str, temperature: float) -> str:
    prompt = DRAFT_PROMPT.format(draft_id=draft_id, context=ctx, query=query)
    try:
        resp = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": temperature},
        )
        return str(resp.get("message", {}).get("content", "")).strip()
    except Exception as e:
        log.warning("speculative.draft.error", id=draft_id, error=str(e))
        return ""


_JSON_RX = re.compile(r"\{.*\}", re.DOTALL)


def _verify(drafts: list[str], query: str, model: str) -> tuple[int, float, str]:
    formatted = "\n\n".join(f"DRAFT {i}:\n{d}" for i, d in enumerate(drafts) if d)
    prompt = VERIFY_PROMPT.format(query=query, drafts=formatted)
    try:
        resp = ollama.generate(
            model=model,
            prompt=prompt,
            format="json",
            options={"temperature": 0.0},
        )
        raw = str(resp.get("response", ""))
    except Exception as e:
        log.warning("speculative.verify.error", error=str(e))
        return 0, 0.0, ""
    try:
        data = json.loads(raw)
    except Exception:
        m = _JSON_RX.search(raw)
        if not m:
            return 0, 0.0, ""
        try:
            data = json.loads(m.group(0))
        except Exception:
            return 0, 0.0, ""
    try:
        best = int(data.get("best", 0))
        score = float(data.get("score", 0))
        rationale = str(data.get("rationale", ""))
        return best, score, rationale
    except (TypeError, ValueError):
        return 0, 0.0, ""


def run_speculative(
    query: str,
    context_chunks: list[str],
    citations: list[Citation],
    cfg: SpeculativeConfig | None = None,
) -> SpeculativeResult:
    """Parallel drafters + verifier. Returns the winning answer."""
    settings = load_settings()
    cfg = cfg or settings.speculative
    drafter = cfg.drafter_model or settings.llm.local_model
    verifier = cfg.verifier_model or settings.llm.local_model
    temperature = settings.llm.temperature

    subsets = _partition(context_chunks, cfg.n_drafts)
    drafts: list[str] = [""] * len(subsets)

    def _worker(i: int, ctx: list[str]) -> None:
        drafts[i] = _draft_one(i, "\n---\n".join(ctx), query, drafter, temperature)

    with ThreadPoolExecutor(max_workers=len(subsets) or 1) as pool:
        list(pool.map(lambda args: _worker(*args), enumerate(subsets)))

    surviving = [d for d in drafts if d]
    if not surviving:
        return SpeculativeResult(
            answer="(no drafter produced a response)",
            best_id=-1,
            verifier_score=0.0,
            rationale="",
            drafts=drafts,
        )
    if len(surviving) == 1:
        return SpeculativeResult(
            answer=surviving[0],
            best_id=drafts.index(surviving[0]),
            verifier_score=0.0,
            rationale="single draft, no verification needed",
            drafts=drafts,
        )

    best, score, rationale = _verify(drafts, query, verifier)
    if best < 0 or best >= len(drafts) or not drafts[best]:
        # verifier picked an empty/invalid draft → fall back to first surviving
        best = drafts.index(surviving[0])
    log.info("speculative.done", best=best, score=score, n_drafts=len(surviving))
    _ = threading  # imported only for parallelism awareness
    return SpeculativeResult(
        answer=drafts[best],
        best_id=best,
        verifier_score=score,
        rationale=rationale,
        drafts=drafts,
    )
