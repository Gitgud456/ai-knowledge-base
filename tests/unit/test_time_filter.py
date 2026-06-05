"""Time-aware retrieval: phrase detection + filter building."""

from __future__ import annotations

import pytest

from akb.retrieve.time_filter import build_time_filter, extract_hint


def test_extract_hint_last_march() -> None:
    assert extract_hint("notes from last March about X") is not None


def test_extract_hint_this_week() -> None:
    assert extract_hint("what did I do this week") is not None


def test_extract_hint_yesterday() -> None:
    assert extract_hint("yesterday's meeting") is not None


def test_extract_hint_none_for_plain_query() -> None:
    assert extract_hint("how does TCP/IP work") is None


def test_build_time_filter_returns_modified_at_range() -> None:
    f, label = build_time_filter("notes from this week about X")
    if f is None:
        pytest.skip("dateparser unavailable in this env")
    assert "modified_at" in f
    bounds = f["modified_at"]
    assert "gte" in bounds and "lte" in bounds
    assert label  # human-readable label populated


def test_build_time_filter_open_ended_since() -> None:
    f, label = build_time_filter("everything since 2024 on topic")
    if f is None:
        pytest.skip("dateparser unavailable in this env")
    assert "modified_at" in f
    assert "gte" in f["modified_at"]
    assert label


def test_build_time_filter_none_for_plain_query() -> None:
    f, label = build_time_filter("how does the agent decide which tool to call")
    assert f is None
    assert label == ""
