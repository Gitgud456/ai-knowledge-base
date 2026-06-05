"""Sidecar SQLite that tracks ``path → content_hash → chunk_ids`` for the
incremental sync loop.

The Qdrant index is *the* source of truth for retrieval. This file is *the*
source of truth for "what's already in there" so the sync loop can do cheap
set arithmetic instead of re-embedding the world every time.

Schema::

    sources(
        source_id   TEXT PRIMARY KEY,
        path        TEXT UNIQUE,
        content_hash TEXT NOT NULL,
        last_seen   TEXT NOT NULL,
        updated_at  TEXT NOT NULL
    )
    chunk_index(
        chunk_id    TEXT PRIMARY KEY,
        source_id   TEXT NOT NULL REFERENCES sources(source_id) ON DELETE CASCADE
    )
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

from akb.config import load_settings
from akb.store.migrations import migrate


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    path: str
    content_hash: str
    last_seen: str
    updated_at: str


@contextmanager
def _conn(path: Path) -> Iterator[sqlite3.Connection]:
    path.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(path)
    c.execute("PRAGMA foreign_keys = ON")
    try:
        yield c
        c.commit()
    finally:
        c.close()


class IngestState:
    def __init__(self, db_path: Path | None = None) -> None:
        self._path = db_path or load_settings().paths.ingest_state_db
        self._init()

    def _init(self) -> None:
        with _conn(self._path) as c:
            c.execute(
                "CREATE TABLE IF NOT EXISTS sources ("
                "source_id TEXT PRIMARY KEY, "
                "path TEXT UNIQUE, "
                "content_hash TEXT NOT NULL, "
                "last_seen TEXT NOT NULL, "
                "updated_at TEXT NOT NULL)"
            )
            c.execute(
                "CREATE TABLE IF NOT EXISTS chunk_index ("
                "chunk_id TEXT PRIMARY KEY, "
                "source_id TEXT NOT NULL REFERENCES sources(source_id) ON DELETE CASCADE)"
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunk_index_source "
                "ON chunk_index(source_id)"
            )
        migrate(self._path, "ingest_state")

    # ---------- queries ----------

    def get(self, source_id: str) -> SourceRecord | None:
        with _conn(self._path) as c:
            row = c.execute(
                "SELECT source_id, path, content_hash, last_seen, updated_at "
                "FROM sources WHERE source_id = ?",
                (source_id,),
            ).fetchone()
        return SourceRecord(*row) if row else None

    def all_source_ids(self) -> set[str]:
        with _conn(self._path) as c:
            rows = c.execute("SELECT source_id FROM sources").fetchall()
        return {r[0] for r in rows}

    def chunk_ids_for(self, source_id: str) -> list[str]:
        with _conn(self._path) as c:
            rows = c.execute(
                "SELECT chunk_id FROM chunk_index WHERE source_id = ?",
                (source_id,),
            ).fetchall()
        return [r[0] for r in rows]

    # ---------- mutations ----------

    def touch(self, source_id: str) -> None:
        now = datetime.now().isoformat()
        with _conn(self._path) as c:
            c.execute("UPDATE sources SET last_seen = ? WHERE source_id = ?", (now, source_id))

    def upsert_source(
        self,
        source_id: str,
        path: str,
        content_hash: str,
        chunk_ids: Iterable[str],
    ) -> None:
        now = datetime.now().isoformat()
        with _conn(self._path) as c:
            c.execute(
                "INSERT INTO sources (source_id, path, content_hash, last_seen, updated_at) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(source_id) DO UPDATE SET "
                "path = excluded.path, "
                "content_hash = excluded.content_hash, "
                "last_seen = excluded.last_seen, "
                "updated_at = excluded.updated_at",
                (source_id, path, content_hash, now, now),
            )
            c.execute("DELETE FROM chunk_index WHERE source_id = ?", (source_id,))
            c.executemany(
                "INSERT INTO chunk_index (chunk_id, source_id) VALUES (?, ?)",
                [(cid, source_id) for cid in chunk_ids],
            )

    def delete_source(self, source_id: str) -> list[str]:
        chunk_ids = self.chunk_ids_for(source_id)
        with _conn(self._path) as c:
            c.execute("DELETE FROM sources WHERE source_id = ?", (source_id,))
            # ON DELETE CASCADE handles chunk_index
        return chunk_ids
