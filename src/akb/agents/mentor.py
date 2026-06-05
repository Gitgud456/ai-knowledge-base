"""Mentor mode.

Three states:
  * ``initial``  — generate a numbered LEARNING PLAN + teach topic 1
  * ``lesson``   — teach the next topic (after a NEXT/BACK intent)
  * ``qa``       — answer a question about the current topic

Intent classification uses the local LLM with a tight 3-class prompt
(replaces the fragile substring-match in the legacy code).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass

import ollama

from akb.agents.tools import search_knowledge_base
from akb.config import load_settings
from akb.prompts.mentor import INTENT__V1, LESSON__V2, PLAN__V2, QA__V2
from akb.schemas import Citation


@dataclass
class MentorState:
    topic: str
    plan: list[str]
    current_index: int = 0
    history: list[dict[str, str]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.history is None:
            self.history = []

    def commit_reply(self, full_text: str) -> "MentorState":
        """Append the assistant's just-emitted reply to history and return self.

        ``start_session`` / ``continue_session`` return a state whose history
        ends on the user turn (because the assistant reply is still streaming
        out). Callers MUST invoke this after fully consuming the stream so the
        next turn has conversational continuity. Without it the model only sees
        a sequence of user messages.
        """
        if full_text:
            self.history.append({"role": "assistant", "content": full_text})
        return self


_PLAN_RX = re.compile(r"LEARNING PLAN:(.*?)(?:\n\n|$)", re.DOTALL | re.IGNORECASE)
_ITEM_RX = re.compile(r"^\s*\d+\.\s*(.+)$", re.MULTILINE)


def parse_plan(text: str) -> list[str]:
    """Extract numbered topics under the LEARNING PLAN: header. Stable across formats."""
    m = _PLAN_RX.search(text)
    if not m:
        return []
    return [item.strip() for item in _ITEM_RX.findall(m.group(1))]


def _classify_intent(message: str) -> str:
    cfg = load_settings().llm
    try:
        resp = ollama.generate(
            model=cfg.local_model,
            prompt=INTENT__V1.format(message=message),
            options={"temperature": 0.0, "num_predict": 8},
        )
        out = str(resp.get("response", "")).strip().upper()
        for tag in ("NEXT", "BACK", "QUESTION"):
            if tag in out:
                return tag
    except Exception:
        pass
    # Fallback to keyword heuristic if LLM hiccups.
    low = message.lower()
    if any(w in low for w in ("next", "proceed", "continue", "move on")):
        return "NEXT"
    if any(w in low for w in ("back", "previous", "go back")):
        return "BACK"
    return "QUESTION"


def start_session(
    topic: str,
    *,
    history: list[dict[str, str]] | None = None,
) -> tuple[Iterator[str], list[Citation], MentorState]:
    """Initial plan + lesson 1, streamed. Returns (stream, citations, state-stub).

    The caller must update ``state.plan`` by calling :func:`parse_plan` on the
    full text after the stream finishes — we can't parse a stream we haven't
    finished emitting.
    """
    cfg = load_settings()
    context, citations = search_knowledge_base(
        topic,
        n_results=cfg.mentor.initial_recall,
        top_k=cfg.mentor.initial_top_k,
    )
    if not context:
        def _no_info() -> Iterator[str]:
            yield "I couldn't find enough material in your vault to build a plan for that topic."
        return _no_info(), [], MentorState(topic=topic, plan=[], history=history or [])

    prompt = PLAN__V2.format(context=context, topic=topic)
    messages = (history or []) + [{"role": "user", "content": prompt}]
    state = MentorState(topic=topic, plan=[], current_index=0, history=list(messages))
    return _stream(cfg.llm.local_model, messages), citations, state


def continue_session(
    state: MentorState,
    user_message: str,
) -> tuple[Iterator[str], list[Citation], MentorState]:
    cfg = load_settings()
    intent = _classify_intent(user_message)
    new_index = state.current_index

    if intent in {"NEXT", "BACK"} and state.plan:
        delta = 1 if intent == "NEXT" else -1
        new_index = max(0, min(state.current_index + delta, len(state.plan) - 1))
        topic = state.plan[new_index]
        context, citations = search_knowledge_base(
            topic,
            n_results=cfg.mentor.topic_recall,
            top_k=cfg.mentor.topic_top_k,
        )
        prompt = LESSON__V2.format(topic=topic, context=context or "(no specific context found)")
    else:
        topic = state.plan[state.current_index] if state.plan else state.topic
        context, citations = search_knowledge_base(
            user_message,
            n_results=cfg.mentor.topic_recall,
            top_k=cfg.mentor.topic_top_k,
        )
        prompt = QA__V2.format(
            topic=topic, context=context or "(no specific context found)", question=user_message
        )

    msgs = state.history[-cfg.mentor.history_window :] + [{"role": "user", "content": prompt}]
    new_state = MentorState(
        topic=state.topic,
        plan=state.plan,
        current_index=new_index,
        history=msgs,
    )
    return _stream(cfg.llm.local_model, msgs), citations, new_state


def _stream(model: str, messages: list[dict[str, str]]) -> Iterator[str]:
    try:
        for chunk in ollama.chat(model=model, messages=messages, stream=True):
            piece = chunk.get("message", {}).get("content", "")
            if piece:
                yield str(piece)
    except Exception as e:
        yield f"\n[mentor stream error: {e}]"


_ = json  # reserved for future structured-output mode
