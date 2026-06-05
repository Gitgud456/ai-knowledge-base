"""Session + message persistence (ported from the legacy ``session_history.db``).

Schema is byte-for-byte the same as the original so the existing DB file works
without migration — only the import path changes.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

from akb.config import load_settings
from akb.store.migrations import migrate


@contextmanager
def _conn(path: Path) -> Iterator[sqlite3.Connection]:
    path.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(path)
    try:
        yield c
        c.commit()
    finally:
        c.close()


def _db_path() -> Path:
    return load_settings().paths.session_db


def init_history_db() -> None:
    with _conn(_db_path()) as c:
        cur = c.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS sessions ("
            "id INTEGER PRIMARY KEY, name TEXT, created_at TEXT)"
        )
        cur.execute(
            "CREATE TABLE IF NOT EXISTS messages ("
            "id INTEGER PRIMARY KEY, session_id INTEGER, role TEXT, "
            "content TEXT, timestamp TEXT, "
            "FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE)"
        )
    migrate(_db_path(), "session_history")


def create_new_session(name: str) -> int:
    with _conn(_db_path()) as c:
        cur = c.cursor()
        cur.execute(
            "INSERT INTO sessions (name, created_at) VALUES (?, ?)",
            (name, datetime.now().isoformat()),
        )
        return int(cur.lastrowid)


def add_message_to_session(session_id: int, role: str, content: str) -> None:
    with _conn(_db_path()) as c:
        c.execute(
            "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (session_id, role, content, datetime.now().isoformat()),
        )


def get_session_history(session_id: int) -> list[dict[str, str]]:
    with _conn(_db_path()) as c:
        rows = c.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY timestamp ASC",
            (session_id,),
        ).fetchall()
    return [{"role": r, "content": ct} for r, ct in rows]


def list_sessions() -> list[dict[str, object]]:
    with _conn(_db_path()) as c:
        rows = c.execute(
            "SELECT id, name, created_at FROM sessions ORDER BY created_at DESC"
        ).fetchall()
    return [{"id": i, "name": n, "created_at": ts} for i, n, ts in rows]


def delete_session(session_id: int) -> None:
    with _conn(_db_path()) as c:
        c.execute("PRAGMA foreign_keys = ON")
        c.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
