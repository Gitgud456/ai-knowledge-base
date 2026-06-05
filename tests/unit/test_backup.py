"""Backup / restore round-trip over a fake data dir."""

from __future__ import annotations

from pathlib import Path

import pytest

from akb import config as akb_config
from akb.ops.backup import backup, list_backups, restore


@pytest.fixture
def fake_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    data = tmp_path / "data"
    data.mkdir()
    (data / "ingest_state.db").write_bytes(b"sqlite-fake-bytes")
    (data / "qdrant").mkdir()
    (data / "qdrant" / "config.json").write_text("{}", encoding="utf-8")
    (data / "context_cache.db").write_bytes(b"cache-fake-bytes")

    # Make load_settings return paths that point at our tmp_path
    settings = akb_config.load_settings()
    monkeypatch.setattr(settings.paths, "data_dir", data, raising=False)
    return data


def test_backup_creates_archive(tmp_path: Path, fake_data: Path) -> None:
    out_dir = tmp_path / "backups"
    info = backup(out_dir=out_dir)
    assert info.path.exists()
    assert info.path.suffix == ".gz"
    assert info.size_bytes > 0


def test_list_backups_finds_archive(tmp_path: Path, fake_data: Path) -> None:
    out_dir = tmp_path / "backups"
    backup(out_dir=out_dir)
    listed = list_backups(out_dir)
    assert len(listed) == 1
    assert listed[0].path.name.startswith("akb-backup-")


def test_round_trip(tmp_path: Path, fake_data: Path) -> None:
    out_dir = tmp_path / "backups"
    info = backup(out_dir=out_dir)

    # Mutate the live data dir so we can verify restore really replaced it
    (fake_data / "ingest_state.db").write_bytes(b"mutated")

    aside = restore(info.path, yes=True)
    assert aside.exists()
    # After restore, the file should contain the original bytes
    assert (fake_data / "ingest_state.db").read_bytes() == b"sqlite-fake-bytes"
    assert (fake_data / "context_cache.db").read_bytes() == b"cache-fake-bytes"
