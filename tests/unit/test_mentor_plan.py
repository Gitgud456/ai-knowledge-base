from __future__ import annotations

from akb.agents.mentor import parse_plan


def test_parse_plan_basic() -> None:
    text = (
        "Some preamble.\n"
        "LEARNING PLAN:\n"
        "1. Basics\n"
        "2. Intermediate stuff\n"
        "3. Advanced topics\n\n"
        "Lesson begins here..."
    )
    assert parse_plan(text) == ["Basics", "Intermediate stuff", "Advanced topics"]


def test_parse_plan_case_insensitive_marker() -> None:
    text = "learning plan:\n1. A\n2. B\n\nbody"
    assert parse_plan(text) == ["A", "B"]


def test_parse_plan_missing_returns_empty() -> None:
    assert parse_plan("just chatting") == []
