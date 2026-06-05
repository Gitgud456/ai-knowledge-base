"""Backup + restore for the entire ``data/`` directory.

Embedded Qdrant stores its collection state on disk under ``data/qdrant/``;
SQLite databases (``ingest_state.db``, ``context_cache.db``,
``session_history.db``) sit next to it. The simplest reliable backup is just
a timestamped tar/zip of the whole ``data/`` dir.

We use ``tarfile`` with gzip — works on every platform, no extra deps. Restore
extracts atomically to a temp location and then swaps to avoid half-restored
state on interrupt.
"""

from __future__ import annotations

import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from akb.config import load_settings
from akb.obs.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class BackupInfo:
    path: Path
    size_bytes: int
    created_at: str


def _data_dir() -> Path:
    return load_settings().paths.data_dir


def backup(out_dir: Path | None = None) -> BackupInfo:
    """Tar+gzip the data dir into ``<out_dir>/akb-backup-<utc>.tar.gz``."""
    data = _data_dir()
    if not data.exists():
        raise RuntimeError(f"data dir does not exist: {data}")

    out_dir = out_dir or (data.parent / "backups")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"akb-backup-{ts}.tar.gz"

    with tarfile.open(out_path, "w:gz") as tf:
        tf.add(data, arcname=data.name)

    size = out_path.stat().st_size
    log.info("backup.done", path=str(out_path), size_bytes=size)
    return BackupInfo(path=out_path, size_bytes=size, created_at=ts)


def restore(archive: Path, *, yes: bool = False) -> Path:
    """Replace the live data dir with the contents of ``archive``.

    The restore happens atomically: extract to a temp dir, then move the live
    data dir aside and swap. A failed extract leaves the existing data dir
    untouched.
    """
    if not archive.exists():
        raise FileNotFoundError(archive)

    data = _data_dir()
    parent = data.parent
    parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="akb-restore-", dir=str(parent)) as tmp:
        tmp_path = Path(tmp)
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(tmp_path)  # nosec: trusted source (our own backup format)

        candidates = [p for p in tmp_path.iterdir() if p.is_dir()]
        if not candidates:
            raise RuntimeError("archive contains no top-level directory")
        source = candidates[0]

        backup_aside = parent / f".data.bak.{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}"
        if data.exists():
            data.rename(backup_aside)
        shutil.move(str(source), str(data))
        log.info(
            "restore.done", archive=str(archive), live=str(data), aside=str(backup_aside)
        )
        return backup_aside


def list_backups(backup_dir: Path | None = None) -> list[BackupInfo]:
    backup_dir = backup_dir or (_data_dir().parent / "backups")
    if not backup_dir.exists():
        return []
    out: list[BackupInfo] = []
    for p in sorted(backup_dir.glob("akb-backup-*.tar.gz")):
        st = p.stat()
        ts = p.stem.removeprefix("akb-backup-")
        out.append(BackupInfo(path=p, size_bytes=st.st_size, created_at=ts))
    return out
