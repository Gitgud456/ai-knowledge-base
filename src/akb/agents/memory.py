"""Conversation memory.

Short-term: a rolling window of recent messages, plus a running summary that
the agent maintains itself when the window overflows. Long-term: each Q/A pair
is persisted to a dedicated Qdrant collection so future turns can recall older
exchanges by similarity (cheap "did we talk about X before?" answers).

Phase 5 wires the short-term piece; the long-term collection is a thin add-on
in the same file so Phase 8 tracing can instrument both with one decorator.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TurnWindow:
    history: list[dict[str, str]]
    summary: str | None = None
    window: int = 8

    def trimmed(self) -> list[dict[str, str]]:
        if len(self.history) <= self.window:
            return list(self.history)
        head: list[dict[str, str]] = []
        if self.summary:
            head.append({"role": "system", "content": f"Conversation summary so far:\n{self.summary}"})
        return head + self.history[-self.window :]

    def append(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})


def trim_history(history: list[dict[str, str]], window: int = 8) -> list[dict[str, str]]:
    """Convenience wrapper used by nodes that don't carry a TurnWindow."""
    return TurnWindow(history=list(history), window=window).trimmed()
