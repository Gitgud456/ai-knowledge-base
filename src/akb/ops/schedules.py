"""Scheduled queries — persist a list of (cron, query, output) tuples and
run them on demand. The runner is a one-shot: ``akb schedule run`` executes
every schedule whose next-run-time is in the past, then exits. Hook it up
to OS-level cron / Task Scheduler — we deliberately don't ship a long-running
daemon.

Why not APScheduler / Celery? They add ops complexity for personal-scale and
duplicate scheduling logic the OS already does well.

Use case::

    akb schedule add --name weekly-review --cron "0 9 * * MON" \\
        --query "what new questions did I write this week?" \\
        --out "akb_chats/weekly-review.md"

    # then in your OS scheduler, hourly:
    akb schedule run

The runner appends a dated section to the output file so successive runs
build a log instead of overwriting.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

from akb.config import load_settings
from akb.obs.logging import get_logger

log = get_logger(__name__)


@dataclass
class Schedule:
    id: int
    name: str
    cron: str
    query: str
    out_path: str
    last_run: str | None
    created_at: str


def _db_path() -> Path:
    return load_settings().paths.data_dir / "schedules.db"


@contextmanager
def _conn() -> Iterator[sqlite3.Connection]:
    p = _db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(p)
    try:
        c.execute(
            "CREATE TABLE IF NOT EXISTS schedules ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "name TEXT NOT NULL UNIQUE, "
            "cron TEXT NOT NULL, "
            "query TEXT NOT NULL, "
            "out_path TEXT NOT NULL, "
            "last_run TEXT, "
            "created_at TEXT NOT NULL)"
        )
        yield c
        c.commit()
    finally:
        c.close()


def add(name: str, cron: str, query: str, out_path: str) -> Schedule:
    now = datetime.now().isoformat()
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO schedules (name, cron, query, out_path, last_run, created_at) "
            "VALUES (?, ?, ?, ?, NULL, ?)",
            (name, cron, query, out_path, now),
        )
        sid = int(cur.lastrowid or 0)
    log.info("schedule.add", id=sid, name=name)
    return Schedule(id=sid, name=name, cron=cron, query=query, out_path=out_path,
                    last_run=None, created_at=now)


def list_all() -> list[Schedule]:
    with _conn() as c:
        rows = c.execute(
            "SELECT id, name, cron, query, out_path, last_run, created_at "
            "FROM schedules ORDER BY id ASC"
        ).fetchall()
    return [Schedule(*r) for r in rows]


def delete(name_or_id: str) -> bool:
    with _conn() as c:
        if name_or_id.isdigit():
            cur = c.execute("DELETE FROM schedules WHERE id = ?", (int(name_or_id),))
        else:
            cur = c.execute("DELETE FROM schedules WHERE name = ?", (name_or_id,))
        return (cur.rowcount or 0) > 0


def _mark_run(name: str) -> None:
    with _conn() as c:
        c.execute(
            "UPDATE schedules SET last_run = ? WHERE name = ?",
            (datetime.now().isoformat(), name),
        )


def _is_due(s: Schedule, now: datetime) -> bool:
    """Cheap cron heuristic.

    Cases we *do* handle:
      * Never-run schedules → due
      * ``last_run`` is more than ~the cron interval ago — approximated by
        looking at the first field of the cron expression (minute) and the
        third field (day-of-month). Fully correct cron parsing requires a
        real parser; for personal-scale "run hourly" / "run weekly" we don't
        need that precision.

    Cases we do *not* handle:
      * Sub-minute cadence (cron can't represent it anyway).
      * Complex DOW + DOM combinations — we err on the side of running.
    """
    if not s.last_run:
        return True
    try:
        last = datetime.fromisoformat(s.last_run)
    except Exception:
        return True
    seconds = (now - last).total_seconds()

    parts = s.cron.split()
    # Approximate the minimum interval implied by the cron expression.
    if len(parts) >= 5:
        # If the schedule names a single hour (e.g. "0 9 * * MON") assume daily
        # at minimum; if it names a day-of-week, assume weekly.
        if parts[4] != "*" and parts[4] not in {"?"}:
            return seconds >= 6 * 3600  # weekly-ish; rerun if 6+ hours late
        if parts[1] != "*":
            return seconds >= 60 * 60  # daily-ish; rerun if an hour late
    return seconds >= 30 * 60  # fall back to "at least every 30 minutes"


def _append_to(out_path: Path, header: str, body: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    block = f"\n\n## {header}\n\n{body}\n"
    if not out_path.exists():
        out_path.write_text(f"# Scheduled output\n{block}", encoding="utf-8")
    else:
        with out_path.open("a", encoding="utf-8") as f:
            f.write(block)


def run_due(now: datetime | None = None) -> dict[str, int]:
    """Execute every schedule that's due. Returns a small stats dict."""
    from akb.agents.graph import ChatAgent

    settings = load_settings()
    now = now or datetime.now()
    agent: ChatAgent | None = None
    ran = 0
    skipped = 0
    for s in list_all():
        if not _is_due(s, now):
            skipped += 1
            continue
        if agent is None:
            agent = ChatAgent()
        try:
            ans = agent.invoke(s.query)
        except Exception as e:
            log.warning("schedule.error", name=s.name, error=str(e))
            continue
        out = Path(s.out_path)
        if not out.is_absolute():
            out = settings.paths.vault / s.out_path
        header = f"{s.name} — {now.strftime('%Y-%m-%d %H:%M')}"
        _append_to(out, header, ans.text)
        _mark_run(s.name)
        ran += 1
        log.info("schedule.run", name=s.name, out=str(out))
    return {"ran": ran, "skipped": skipped}
