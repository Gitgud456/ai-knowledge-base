"""Scheduled queries CRUD + due-detection heuristics."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from akb import config as akb_config
from akb.ops import schedules


@pytest.fixture(autouse=True)
def _redirect_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = akb_config.load_settings()
    monkeypatch.setattr(settings.paths, "data_dir", tmp_path, raising=False)


def test_add_list_delete() -> None:
    s = schedules.add(name="weekly", cron="0 9 * * MON", query="hi", out_path="x.md")
    assert s.id > 0
    items = schedules.list_all()
    assert len(items) == 1
    assert items[0].name == "weekly"
    assert schedules.delete("weekly") is True
    assert schedules.list_all() == []


def test_due_when_never_run() -> None:
    s = schedules.Schedule(
        id=1, name="n", cron="0 9 * * MON", query="q", out_path="o",
        last_run=None, created_at="2026-01-01T00:00:00",
    )
    assert schedules._is_due(s, datetime.now()) is True


def test_not_due_within_hour_for_daily() -> None:
    now = datetime.now()
    s = schedules.Schedule(
        id=1, name="n", cron="0 9 * * *", query="q", out_path="o",
        last_run=(now - timedelta(minutes=10)).isoformat(),
        created_at=now.isoformat(),
    )
    assert schedules._is_due(s, now) is False


def test_due_when_stale() -> None:
    now = datetime.now()
    s = schedules.Schedule(
        id=1, name="n", cron="0 9 * * *", query="q", out_path="o",
        last_run=(now - timedelta(hours=2)).isoformat(),
        created_at=now.isoformat(),
    )
    assert schedules._is_due(s, now) is True


def test_weekly_threshold_uses_6_hours() -> None:
    now = datetime.now()
    s = schedules.Schedule(
        id=1, name="n", cron="0 9 * * MON", query="q", out_path="o",
        last_run=(now - timedelta(hours=2)).isoformat(),
        created_at=now.isoformat(),
    )
    assert schedules._is_due(s, now) is False
    s = schedules.Schedule(
        id=1, name="n", cron="0 9 * * MON", query="q", out_path="o",
        last_run=(now - timedelta(hours=8)).isoformat(),
        created_at=now.isoformat(),
    )
    assert schedules._is_due(s, now) is True
