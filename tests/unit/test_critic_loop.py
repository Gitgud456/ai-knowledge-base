"""``_route_from_critic`` bound semantics — was off-by-two."""

from __future__ import annotations

import pytest

from akb.agents.graph import _route_from_critic


@pytest.mark.parametrize(
    "verdict,iterations,max_iter,expected",
    [
        # verdict=good always finalizes regardless of iterations
        ("good", 0, 1, "finalize"),
        ("good", 5, 1, "finalize"),
        # verdict=revise + within budget -> retrieve_again
        ("revise", 0, 1, "retrieve_again"),
        ("revise", 1, 1, "retrieve_again"),
        # verdict=revise + exhausted -> finalize
        ("revise", 2, 1, "finalize"),
        # max_critic_iterations=0 means "never revise"
        ("revise", 1, 0, "finalize"),
        ("revise", 0, 0, "retrieve_again"),
        # max_critic_iterations=2 allows two revise loops
        ("revise", 1, 2, "retrieve_again"),
        ("revise", 2, 2, "retrieve_again"),
        ("revise", 3, 2, "finalize"),
    ],
)
def test_critic_boundary(
    monkeypatch: pytest.MonkeyPatch,
    verdict: str,
    iterations: int,
    max_iter: int,
    expected: str,
) -> None:
    from akb import config as akb_config

    class _FakeAgent:
        max_critic_iterations = max_iter

    class _FakeSettings:
        agent = _FakeAgent()

    monkeypatch.setattr(akb_config, "load_settings", lambda: _FakeSettings())
    # graph.py captured load_settings via `from akb.config import load_settings`,
    # so patch there too:
    from akb.agents import graph as g

    monkeypatch.setattr(g, "load_settings", lambda: _FakeSettings())

    assert _route_from_critic({"critic_verdict": verdict, "iterations": iterations}) == expected
