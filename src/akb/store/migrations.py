"""SQLite schema versioning + Qdrant index-version sentinel.

Two distinct guarantees:

1. **SQLite migrations.** Every akb-owned SQLite file (``ingest_state``,
   ``context_cache``, ``session_history``) carries a single-row
   ``schema_version(version INTEGER)`` table. :func:`migrate` runs the right
   step functions in order and updates the row atomically. Adding a new
   migration is two lines below.

2. **Qdrant index stamp.** The vector store records ``(embed_model, embed_dim,
   binary_quantization, akb_version)`` in a designated collection
   (``akb_meta``) at first write. On startup :func:`check_index_compatible`
   refuses to proceed if the live config disagrees — without a
   ``--force-reindex`` flag the user would silently embed with one model and
   query with another. The check is best-effort; a missing stamp is treated
   as "first run" and the current config is written through.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from akb import __version__
from akb.config import EmbedConfig
from akb.obs.logging import get_logger

log = get_logger(__name__)

INDEX_META_COLLECTION = "akb_meta"


# --------- SQLite migrations -----------------------------------------------


@contextmanager
def _conn(path: Path) -> Iterator[sqlite3.Connection]:
    path.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(path)
    try:
        yield c
        c.commit()
    finally:
        c.close()


def _current_version(c: sqlite3.Connection) -> int:
    c.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
    row = c.execute("SELECT version FROM schema_version").fetchone()
    if row is None:
        c.execute("INSERT INTO schema_version (version) VALUES (0)")
        return 0
    return int(row[0])


def _set_version(c: sqlite3.Connection, v: int) -> None:
    c.execute("UPDATE schema_version SET version = ?", (v,))


# Per-database migration tables. Each entry is (target_version, fn).
# Migration functions must be IDEMPOTENT — they may run after partial earlier
# runs. They take a connection inside an open transaction.
MIGRATIONS: dict[str, list[tuple[int, Callable[[sqlite3.Connection], None]]]] = {
    "ingest_state": [],
    "context_cache": [],
    "session_history": [],
}


def migrate(db_path: Path, kind: str) -> int:
    """Bring ``db_path`` up to the latest schema for the given ``kind``.

    Returns the resulting schema_version. No-op for unknown kinds.
    """
    if kind not in MIGRATIONS:
        return 0
    with _conn(db_path) as c:
        cur = _current_version(c)
        steps = sorted(MIGRATIONS[kind])
        for target, fn in steps:
            if target <= cur:
                continue
            log.info("migrate.run", kind=kind, from_=cur, to=target)
            fn(c)
            _set_version(c, target)
            cur = target
        return cur


# --------- Qdrant index stamp ----------------------------------------------


@dataclass(frozen=True)
class IndexStamp:
    embed_model: str
    embed_dim: int
    binary_quantization: bool
    akb_version: str

    @classmethod
    def from_payload(cls, p: dict[str, object]) -> "IndexStamp":
        return cls(
            embed_model=str(p.get("embed_model", "")),
            embed_dim=int(p.get("embed_dim", 0) or 0),
            binary_quantization=bool(p.get("binary_quantization", False)),
            akb_version=str(p.get("akb_version", "")),
        )


def _stamp_from_config(cfg: EmbedConfig) -> IndexStamp:
    return IndexStamp(
        embed_model=cfg.model,
        embed_dim=cfg.dim,
        binary_quantization=cfg.binary_quantization,
        akb_version=__version__,
    )


def _ensure_meta_collection(client: object) -> None:
    from qdrant_client import models  # type: ignore[import-untyped]

    cl = client  # type: ignore[assignment]
    if cl.collection_exists(INDEX_META_COLLECTION):  # type: ignore[attr-defined]
        return
    # Minimal 1-d collection — we just want a point with a payload.
    cl.create_collection(  # type: ignore[attr-defined]
        collection_name=INDEX_META_COLLECTION,
        vectors_config={"v": models.VectorParams(size=1, distance=models.Distance.COSINE)},
    )


def read_index_stamp(client: object) -> IndexStamp | None:
    """Return the live stamp, or ``None`` if no stamp has been written yet."""
    cl = client  # type: ignore[assignment]
    if not cl.collection_exists(INDEX_META_COLLECTION):  # type: ignore[attr-defined]
        return None
    points, _ = cl.scroll(  # type: ignore[attr-defined]
        collection_name=INDEX_META_COLLECTION,
        with_payload=True,
        with_vectors=False,
        limit=1,
    )
    if not points:
        return None
    payload = points[0].payload or {}
    return IndexStamp.from_payload(payload)


def write_index_stamp(client: object, cfg: EmbedConfig) -> IndexStamp:
    from qdrant_client import models  # type: ignore[import-untyped]

    _ensure_meta_collection(client)
    stamp = _stamp_from_config(cfg)
    cl = client  # type: ignore[assignment]
    cl.upsert(  # type: ignore[attr-defined]
        collection_name=INDEX_META_COLLECTION,
        points=[
            models.PointStruct(
                id=1,
                vector={"v": [0.0]},
                payload={
                    "embed_model": stamp.embed_model,
                    "embed_dim": stamp.embed_dim,
                    "binary_quantization": stamp.binary_quantization,
                    "akb_version": stamp.akb_version,
                },
            )
        ],
        wait=True,
    )
    return stamp


@dataclass(frozen=True)
class CompatResult:
    compatible: bool
    reason: str = ""
    live: IndexStamp | None = None
    expected: IndexStamp | None = None


def check_index_compatible(client: object, cfg: EmbedConfig) -> CompatResult:
    """Verify the live Qdrant stamp matches the current config.

    Returns ``compatible=True`` and writes the stamp through on first run
    (no existing stamp). Returns ``compatible=False`` with a human-readable
    reason whenever a *meaningful* mismatch is found (model name / dim /
    quantization). The akb version is logged but does not block.
    """
    expected = _stamp_from_config(cfg)
    live = read_index_stamp(client)
    if live is None:
        write_index_stamp(client, cfg)
        log.info("index.stamp.write", **expected.__dict__)
        return CompatResult(compatible=True, live=expected, expected=expected)
    if live.embed_model != expected.embed_model:
        return CompatResult(
            compatible=False,
            reason=(
                f"embed model mismatch: index was built with '{live.embed_model}' but config "
                f"says '{expected.embed_model}'. Reindex required."
            ),
            live=live,
            expected=expected,
        )
    if live.embed_dim != expected.embed_dim:
        return CompatResult(
            compatible=False,
            reason=(
                f"embed dim mismatch: index dim={live.embed_dim} vs config dim={expected.embed_dim}."
            ),
            live=live,
            expected=expected,
        )
    if live.binary_quantization != expected.binary_quantization:
        return CompatResult(
            compatible=False,
            reason=(
                f"binary_quantization mismatch: index={live.binary_quantization} vs config="
                f"{expected.binary_quantization}. Reindex required."
            ),
            live=live,
            expected=expected,
        )
    if live.akb_version != expected.akb_version:
        log.info(
            "index.stamp.akb_version_diff",
            live=live.akb_version,
            expected=expected.akb_version,
        )
    return CompatResult(compatible=True, live=live, expected=expected)
