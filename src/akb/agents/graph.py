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

The router can be **bypassed by slash commands** (``/search``, ``/web``,
``/cite``, ``/dry-run``) — see :mod:`akb.agents.slash`. Pinned ``[[Note]]``
mentions in the query are pre-resolved by :mod:`akb.agents.pinning` and
prepended to the retrieved context.

State is a TypedDict that LangGraph reduces. We persist nothing in this file —
session storage stays in :mod:`akb.sessions.db`.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

import ollama
from langgraph.graph import END, START, StateGraph

from akb.agents.memory import trim_history
from akb.agents.pinning import (
    Pinned,
    extract_mentions,
    format_pinned_block,
    resolve as resolve_pins,
    strip_mentions,
)
from akb.agents.tools import search_knowledge_base, search_web
from akb.config import load_settings
from akb.ingest.graph import VaultGraph
from akb.obs.logging import get_logger
from akb.prompts.chat import CRITIC__V1, DIRECT__V1, DRAFT__V1, FINAL__V1, ROUTER__V1
from akb.schemas import Answer, Citation

log = get_logger(__name__)


class ChatState(TypedDict, total=False):
    query: str
    raw_query: str            # original (pre-mention-strip) text
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
    sub_queries: list[str]
    pinned_titles: list[str]
    # Slash-command knobs
    force_path: str           # 'retrieve' | 'web' | 'direct' (overrides router)
    cite_only: bool           # short-circuit: return citations + tiny preamble
    dry_run: bool             # stop after retrieve; surface what would be sent


def _route(state: ChatState) -> ChatState:
    cfg = load_settings()
    query = state["query"]
    # Slash command override
    forced = state.get("force_path")
    if forced in {"retrieve", "web", "direct"}:
        log.info("agent.route", path=forced, forced=True)
        return {**state, "path": forced}
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

    # Prepend any pinned-note blocks the UI resolved before invoke()
    pinned_block = state.get("context", "") if state.get("pinned_titles") else ""
    if pinned_block:
        context = f"{pinned_block}\n---\n{context}" if context else pinned_block
    return {**state, "context": context, "citations": citations}


def _retrieve_web(state: ChatState) -> ChatState:
    query = state.get("improved_query") or state["query"]
    if not load_settings().agent.enable_web_fallback:
        return {**state, "context": "", "citations": []}
    return {**state, "context": search_web(query), "citations": []}


def _draft(state: ChatState) -> ChatState:
    if state.get("dry_run") or state.get("cite_only"):
        # Skip the draft entirely on dry-run / cite-only — nothing to draft.
        return {**state, "draft": ""}
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
    if state.get("dry_run") or state.get("cite_only"):
        return {**state, "critic_verdict": "good", "critic_notes": ""}
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
    """Synchronous finalize used by ``invoke``. ``stream_answer`` calls
    :func:`stream_finalize_messages` directly for token streaming."""
    if state.get("dry_run"):
        return {**state, "final": _format_dry_run(state)}
    if state.get("cite_only"):
        return {**state, "final": _format_cite_only(state)}
    cfg = load_settings()
    messages = _final_messages(state, cfg)
    try:
        resp = ollama.chat(
            model=cfg.llm.local_model,
            messages=messages,
            options={"temperature": cfg.llm.temperature},
        )
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


def _final_messages(state: ChatState, cfg: Any) -> list[dict[str, str]]:
    prompt = FINAL__V1.format(
        context=state.get("context", ""),
        query=state["query"],
        draft=state.get("draft", ""),
        notes=state.get("critic_notes", ""),
    )
    history = trim_history(state.get("history", []), window=cfg.agent.history_window)
    return [*history, {"role": "user", "content": prompt}]


def _direct_messages(state: ChatState, cfg: Any) -> list[dict[str, str]]:
    prompt = DIRECT__V1.format(query=state["query"])
    history = trim_history(state.get("history", []), window=cfg.agent.history_window)
    return [*history, {"role": "user", "content": prompt}]


def _format_cite_only(state: ChatState) -> str:
    cits = state.get("citations") or []
    if not cits:
        return "No relevant chunks found."
    lines = [f"Top {len(cits)} chunk(s):", ""]
    for i, c in enumerate(cits, 1):
        snippet = c.snippet.replace("\n", " ")
        lines.append(f"{i}. `{c.source_id}` (score {c.score:.3f})\n    > {snippet}")
    return "\n".join(lines)


def _format_dry_run(state: ChatState) -> str:
    cits = state.get("citations") or []
    return (
        "**Dry run** — retrieval ran, no synthesis was performed.\n\n"
        f"path: `{state.get('path')}`  iterations: {state.get('iterations', 0)}\n\n"
        f"{len(cits)} chunk(s) would have been sent to the LLM:\n\n"
        + "\n".join(f"- `{c.source_id}` (score {c.score:.3f})" for c in cits)
    )


@dataclass
class StreamEvent:
    """One event from :meth:`ChatAgent.stream_answer`."""

    kind: Literal["state", "token", "done"]
    state: ChatState | None = None
    token: str = ""
    answer: Answer | None = None


@dataclass
class ChatAgent:
    """Compiled LangGraph chat agent. Hold one per process — the compile is cheap."""

    graph: VaultGraph | None = None
    store: Any = None
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

    # ------------------------------------------------------------ helpers

    def _resolve_pinned(self, query: str) -> tuple[str, str, list[Pinned]]:
        """Extract ``[[Note]]`` mentions, fetch their chunks, return
        (stripped_query, pinned_block, pinned_list)."""
        mentions = extract_mentions(query)
        if not mentions:
            return query, "", []
        if self.store is None:
            # Lazy import to avoid pulling Qdrant in unit tests that don't need it
            from akb.store.qdrant_store import get_store

            store = get_store()
        else:
            store = self.store
        pinned = resolve_pins(mentions, self.graph, store)
        stripped = strip_mentions(query)
        return stripped, format_pinned_block(pinned), pinned

    def _init_state(
        self,
        query: str,
        history: list[dict[str, str]] | None,
        *,
        force_path: str | None = None,
        cite_only: bool = False,
        dry_run: bool = False,
    ) -> ChatState:
        stripped, pinned_block, pinned = self._resolve_pinned(query)
        state: ChatState = {
            "query": stripped,
            "raw_query": query,
            "history": history or [],
            "iterations": 0,
            "pinned_titles": [p.title for p in pinned],
        }
        if pinned_block:
            # Stash the pinned block in `context` so the retrieve_kb node can
            # find it (the node prepends it when pinned_titles is non-empty).
            state["context"] = pinned_block
        if force_path:
            state["force_path"] = force_path
        if cite_only:
            state["cite_only"] = True
        if dry_run:
            state["dry_run"] = True
        return state

    # ------------------------------------------------------------ public API

    def invoke(
        self,
        query: str,
        history: list[dict[str, str]] | None = None,
        *,
        force_path: str | None = None,
        cite_only: bool = False,
        dry_run: bool = False,
    ) -> Answer:
        cfg = load_settings()
        init = self._init_state(
            query, history, force_path=force_path, cite_only=cite_only, dry_run=dry_run
        )
        out: ChatState = self.app.invoke(init)
        return Answer(
            text=out.get("final", ""),
            citations=out.get("citations", []),
            used_tools=[out.get("path", "retrieve")],
            model=cfg.llm.local_model,
            iterations=out.get("iterations", 1),
        )

    def stream(self, query: str, history: list[dict[str, str]] | None = None) -> Iterator[ChatState]:
        for event in self.app.stream(self._init_state(query, history)):
            yield event  # type: ignore[misc]

    async def astream(self, query: str, history: list[dict[str, str]] | None = None) -> AsyncIterator[ChatState]:
        async for event in self.app.astream(self._init_state(query, history)):
            yield event  # type: ignore[misc]

    def stream_answer(
        self,
        query: str,
        history: list[dict[str, str]] | None = None,
        *,
        force_path: str | None = None,
        cite_only: bool = False,
        dry_run: bool = False,
    ) -> Iterator[StreamEvent]:
        """Run the full graph but emit tokens from the *final* LLM call live.

        Sequence::

            -> StreamEvent(kind='state', state=<pre-final state>)
            -> StreamEvent(kind='token', token='...')   (many)
            -> StreamEvent(kind='done',  answer=<Answer>)

        The 'state' event arrives once everything upstream has run (router,
        retrieve, draft, critic). The 'token' events stream from the synthesis
        LLM. 'done' carries the final assembled :class:`Answer`.

        For ``dry_run`` and ``cite_only``, the synthesis step is skipped and the
        pre-formatted preview/citation block is emitted as a single 'token'.
        """
        cfg = load_settings()
        init = self._init_state(
            query, history, force_path=force_path, cite_only=cite_only, dry_run=dry_run
        )

        # Drive the graph synchronously and capture the last state. We rely on
        # the fact that `app.stream(stream_mode='values')` yields the full state
        # after each node — taking the final one before the finalize/direct
        # node would have run lets us emit tokens ourselves.
        last_state: ChatState = init
        path: str = ""
        for chunk in self.app.stream(init, stream_mode="updates"):
            for node, partial in chunk.items():
                if not partial:
                    continue
                last_state = {**last_state, **partial}
                if node == "router":
                    path = last_state.get("path", "")
                # Stop before the synthesizer runs — we'll do that ourselves
                # for the token stream.
                if node == "critic" and not (
                    last_state.get("dry_run") or last_state.get("cite_only")
                ):
                    if _route_from_critic(last_state) == "finalize":
                        break
            # break outer loop too once we hit the finalize boundary
            if last_state.get("critic_verdict") in {"good"} or (
                last_state.get("iterations", 0) > cfg.agent.max_critic_iterations
            ):
                if last_state.get("path") in {"retrieve", "web"}:
                    break

        yield StreamEvent(kind="state", state=last_state)

        # Cite-only / dry-run short-circuit
        if last_state.get("dry_run"):
            text = _format_dry_run(last_state)
            yield StreamEvent(kind="token", token=text)
            yield StreamEvent(
                kind="done",
                answer=Answer(
                    text=text,
                    citations=last_state.get("citations", []),
                    used_tools=[last_state.get("path", "retrieve")],
                    model=cfg.llm.local_model,
                    iterations=last_state.get("iterations", 0),
                ),
            )
            return
        if last_state.get("cite_only"):
            text = _format_cite_only(last_state)
            yield StreamEvent(kind="token", token=text)
            yield StreamEvent(
                kind="done",
                answer=Answer(
                    text=text,
                    citations=last_state.get("citations", []),
                    used_tools=[last_state.get("path", "retrieve")],
                    model=cfg.llm.local_model,
                    iterations=last_state.get("iterations", 0),
                ),
            )
            return

        # Stream the synthesizer (or the 'direct' path)
        path = last_state.get("path", "retrieve")
        if path == "direct":
            messages = _direct_messages(last_state, cfg)
        else:
            messages = _final_messages(last_state, cfg)
        buf: list[str] = []
        try:
            for chunk in ollama.chat(
                model=cfg.llm.local_model,
                messages=messages,
                stream=True,
                options={"temperature": cfg.llm.temperature},
            ):
                piece = chunk.get("message", {}).get("content", "")
                if not piece:
                    continue
                buf.append(str(piece))
                yield StreamEvent(kind="token", token=str(piece))
        except Exception as e:
            yield StreamEvent(kind="token", token=f"\n[stream error: {e}]")

        final_text = "".join(buf).strip()
        yield StreamEvent(
            kind="done",
            answer=Answer(
                text=final_text,
                citations=last_state.get("citations", []),
                used_tools=[last_state.get("path", "retrieve")],
                model=cfg.llm.local_model,
                iterations=last_state.get("iterations", 1),
            ),
        )
