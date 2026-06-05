"""LangGraph chat agent.

Topology
--------
    START
      │
    router  ────────── "direct" ────── direct_answer ─── END
      │ "retrieve"                          ▲
      ▼                                     │ (good)
    retrieve_kb ──► draft ──► critic ──────►│
      ▲                          │ (revise: improved_query)
      │                          ▼
      └────────────── (loop ≤ max_critic_iterations) ─────
      │ "web"
      ▼
    retrieve_web ──► draft ──► critic ──► END

State is a TypedDict that LangGraph reduces. We persist nothing in this file —
session storage stays in :mod:`akb.sessions.db`.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, TypedDict

import ollama
from langgraph.graph import END, START, StateGraph

from akb.agents.memory import trim_history
from akb.agents.tools import search_knowledge_base, search_web
from akb.config import load_settings
from akb.ingest.graph import VaultGraph
from akb.obs.logging import get_logger
from akb.prompts.chat import CRITIC__V1, DIRECT__V1, DRAFT__V1, FINAL__V1, ROUTER__V1
from akb.schemas import Answer, Citation

log = get_logger(__name__)


class ChatState(TypedDict, total=False):
    query: str
    history: list[dict[str, str]]
    path: Literal["retrieve", "web", "direct"]
    context: str
    citations: list[Citation]
    draft: str
    critic_verdict: Literal["good", "revise"]
    critic_notes: str
    improved_query: str
    iterations: int
    final: str
    deep_mode: bool


def _route(state: ChatState) -> ChatState:
    cfg = load_settings()
    query = state["query"]
    if not cfg.agent.enable_router:
        return {**state, "path": "retrieve"}
    try:
        resp = ollama.generate(
            model=cfg.llm.local_model,
            prompt=ROUTER__V1.format(query=query),
            format="json",
            options={"temperature": 0.0},
        )
        data = json.loads(str(resp.get("response", "{}")))
        path = data.get("path", "retrieve")
        if path not in {"retrieve", "web", "direct"}:
            path = "retrieve"
    except Exception as e:
        log.warning("agent.route.fallback", error=str(e))
        path = "retrieve"
    log.info("agent.route", path=path)
    return {**state, "path": path}


def _retrieve_kb(state: ChatState, *, graph: VaultGraph | None = None) -> ChatState:
    query = state.get("improved_query") or state["query"]
    context, citations = search_knowledge_base(query, graph=graph)
    return {**state, "context": context, "citations": citations}


def _retrieve_web(state: ChatState) -> ChatState:
    query = state.get("improved_query") or state["query"]
    if not load_settings().agent.enable_web_fallback:
        return {**state, "context": "", "citations": []}
    return {**state, "context": search_web(query), "citations": []}


def _draft(state: ChatState) -> ChatState:
    cfg = load_settings()
    context = state.get("context") or "(no context retrieved)"
    prompt = DRAFT__V1.format(context=context, query=state["query"])
    history = trim_history(state.get("history", []), window=cfg.agent.history_window)
    messages = [*history, {"role": "user", "content": prompt}]
    try:
        resp = ollama.chat(
            model=cfg.llm.local_model,
            messages=messages,
            options={"temperature": cfg.llm.temperature},
        )
        draft = str(resp.get("message", {}).get("content", "")).strip()
    except Exception as e:
        draft = f"(draft generation failed: {e})"
    return {**state, "draft": draft}


def _critic(state: ChatState) -> ChatState:
    cfg = load_settings()
    if not cfg.agent.enable_critic:
        return {**state, "critic_verdict": "good", "critic_notes": ""}
    prompt = CRITIC__V1.format(
        context=state.get("context", ""),
        query=state["query"],
        draft=state.get("draft", ""),
    )
    try:
        resp = ollama.generate(
            model=cfg.llm.local_model,
            prompt=prompt,
            format="json",
            options={"temperature": 0.0},
        )
        data = json.loads(str(resp.get("response", "{}")))
        verdict = data.get("verdict", "good")
        notes = data.get("notes", "")
        improved = data.get("improved_query") or state["query"]
    except Exception:
        verdict, notes, improved = "good", "", state["query"]
    iterations = state.get("iterations", 0) + 1
    log.info("agent.critic", verdict=verdict, iterations=iterations)
    return {
        **state,
        "critic_verdict": verdict,
        "critic_notes": notes,
        "improved_query": improved,
        "iterations": iterations,
    }


def _finalize(state: ChatState) -> ChatState:
    cfg = load_settings()
    prompt = FINAL__V1.format(
        context=state.get("context", ""),
        query=state["query"],
        draft=state.get("draft", ""),
        notes=state.get("critic_notes", ""),
    )
    history = trim_history(state.get("history", []), window=cfg.agent.history_window)
    messages = [*history, {"role": "user", "content": prompt}]
    try:
        resp = ollama.chat(model=cfg.llm.local_model, messages=messages,
                            options={"temperature": cfg.llm.temperature})
        final = str(resp.get("message", {}).get("content", "")).strip()
    except Exception as e:
        final = state.get("draft") or f"(finalize failed: {e})"
    return {**state, "final": final}


def _direct(state: ChatState) -> ChatState:
    cfg = load_settings()
    prompt = DIRECT__V1.format(query=state["query"])
    history = trim_history(state.get("history", []), window=cfg.agent.history_window)
    messages = [*history, {"role": "user", "content": prompt}]
    try:
        resp = ollama.chat(model=cfg.llm.local_model, messages=messages,
                            options={"temperature": cfg.llm.temperature})
        final = str(resp.get("message", {}).get("content", "")).strip()
    except Exception as e:
        final = f"(direct generation failed: {e})"
    return {**state, "final": final}


def _route_from_router(state: ChatState) -> str:
    return state.get("path", "retrieve")


def _route_from_critic(state: ChatState) -> str:
    """Decide whether to revise (re-retrieve with improved_query) or finalize.

    Bound semantics: ``max_critic_iterations`` is the count of *revise* loops we
    allow. With the default (1), the path is at most:
      retrieve → draft → critic("revise") → retrieve → draft → critic → finalize
    i.e. one revise. ``max_critic_iterations=0`` short-circuits any revise.
    """
    cfg = load_settings().agent
    verdict = state.get("critic_verdict", "good")
    if verdict == "good":
        return "finalize"
    if state.get("iterations", 0) > cfg.max_critic_iterations:
        return "finalize"
    return "retrieve_again"


@dataclass
class ChatAgent:
    """Compiled LangGraph chat agent. Hold one per process — the compile is cheap."""

    graph: VaultGraph | None = None
    _app: Any = field(default=None, init=False, repr=False)

    def _build(self) -> Any:
        g = StateGraph(ChatState)
        g.add_node("router", _route)
        g.add_node("retrieve_kb", lambda s: _retrieve_kb(s, graph=self.graph))
        g.add_node("retrieve_web", _retrieve_web)
        g.add_node("draft", _draft)
        g.add_node("critic", _critic)
        g.add_node("finalize", _finalize)
        g.add_node("direct", _direct)

        g.add_edge(START, "router")
        g.add_conditional_edges(
            "router",
            _route_from_router,
            {"retrieve": "retrieve_kb", "web": "retrieve_web", "direct": "direct"},
        )
        g.add_edge("retrieve_kb", "draft")
        g.add_edge("retrieve_web", "draft")
        g.add_edge("draft", "critic")
        g.add_conditional_edges(
            "critic",
            _route_from_critic,
            {"finalize": "finalize", "retrieve_again": "retrieve_kb"},
        )
        g.add_edge("finalize", END)
        g.add_edge("direct", END)
        return g.compile()

    @property
    def app(self) -> Any:
        if self._app is None:
            self._app = self._build()
        return self._app

    def invoke(self, query: str, history: list[dict[str, str]] | None = None) -> Answer:
        cfg = load_settings()
        init: ChatState = {"query": query, "history": history or [], "iterations": 0}
        out: ChatState = self.app.invoke(init)
        return Answer(
            text=out.get("final", ""),
            citations=out.get("citations", []),
            used_tools=[out.get("path", "retrieve")],
            model=cfg.llm.local_model,
            iterations=out.get("iterations", 1),
        )

    def stream(self, query: str, history: list[dict[str, str]] | None = None) -> Iterator[ChatState]:
        init: ChatState = {"query": query, "history": history or [], "iterations": 0}
        for event in self.app.stream(init):
            yield event  # type: ignore[misc]

    async def astream(self, query: str, history: list[dict[str, str]] | None = None) -> AsyncIterator[ChatState]:
        init: ChatState = {"query": query, "history": history or [], "iterations": 0}
        async for event in self.app.astream(init):
            yield event  # type: ignore[misc]
