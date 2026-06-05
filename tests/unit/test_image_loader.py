"""Markdown image discovery — both ``![[image]]`` and ``![alt](path)``."""

from __future__ import annotations

from pathlib import Path

import pytest

from akb import config as akb_config
from akb.config import reset_settings_cache
from akb.ingest.image_loader import (
    MD_IMAGE_RX,
    WIKILINK_IMAGE_RX,
    _vault_image_index,
    discover_images,
)


def test_wikilink_image_pattern_matches() -> None:
    m = WIKILINK_IMAGE_RX.search("see ![[diagram.png]] above")
    assert m is not None
    assert m.group(1) == "diagram.png"


def test_wikilink_image_with_alias() -> None:
    m = WIKILINK_IMAGE_RX.search("see ![[diagram.png|the diagram]]")
    assert m is not None
    assert m.group(1) == "diagram.png"


def test_markdown_image_pattern_matches() -> None:
    m = MD_IMAGE_RX.search("inline ![alt](path/to/img.jpg)")
    assert m is not None
    assert m.group(1) == "path/to/img.jpg"


def test_markdown_image_pattern_ignores_links() -> None:
    # `[text](url)` without leading `!` is a normal link, not an image
    assert MD_IMAGE_RX.search("[some link](https://x.com)") is None


@pytest.fixture
def vault(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "attachments").mkdir()
    (tmp_path / "attachments" / "diagram.png").write_bytes(b"")
    (tmp_path / "attachments" / "Screenshot.JPG").write_bytes(b"")
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "buried.png").write_bytes(b"")
    (tmp_path / "Notes.md").write_text(
        "intro\n\n![[diagram.png]]\n\nand ![inline](attachments/Screenshot.JPG)\n",
        encoding="utf-8",
    )
    (tmp_path / "Other.md").write_text("![[buried.png|caption]]", encoding="utf-8")

    monkeypatch.setenv("AKB_PATHS__VAULT", str(tmp_path))
    reset_settings_cache()
    return tmp_path


def test_index_finds_all_image_types(vault: Path) -> None:
    from akb.config import IngestConfig

    idx = _vault_image_index(vault, IngestConfig().attachment_dirs)
    assert "diagram.png" in idx
    assert "screenshot.jpg" in idx
    assert "buried.png" in idx


def test_discover_images_finds_both_syntaxes(vault: Path) -> None:
    refs = discover_images(vault)
    paths = {r.image_path.name.lower() for r in refs}
    assert "diagram.png" in paths
    assert "screenshot.jpg" in paths
    assert "buried.png" in paths


def test_discover_images_dedupes(vault: Path) -> None:
    # Add a second note that points at the same diagram — should still
    # appear only once in the result.
    note2 = vault / "DupeRef.md"
    note2.write_text("![[diagram.png]]", encoding="utf-8")
    refs = discover_images(vault)
    diagrams = [r for r in refs if r.image_path.name == "diagram.png"]
    assert len(diagrams) == 1
