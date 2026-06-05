"""Mention extraction + stripping for ``[[Note]]`` pinning."""

from __future__ import annotations

from akb.agents.pinning import extract_mentions, strip_mentions


def test_extract_single() -> None:
    assert extract_mentions("see [[Project Plan]] please") == ["Project Plan"]


def test_extract_alias_and_heading() -> None:
    assert extract_mentions("look at [[Notes#Section|alias]] for more") == ["Notes"]


def test_extract_dedup() -> None:
    out = extract_mentions("[[A]] and [[A]] and [[B]]")
    assert out == ["A", "B"]


def test_strip_keeps_visible_text() -> None:
    assert strip_mentions("see [[Project Plan]] please") == "see Project Plan please"
    assert strip_mentions("look at [[Notes#Section|aka X]] now") == "look at aka X now"


def test_strip_no_mentions() -> None:
    assert strip_mentions("just text") == "just text"
