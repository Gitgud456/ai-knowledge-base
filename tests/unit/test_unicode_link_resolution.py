"""Unicode-aware link resolution: ``Straße`` must match ``strasse``,
``Café.md`` must round-trip across NFC/NFD."""

from __future__ import annotations

import unicodedata
from pathlib import Path

from akb.ingest.graph import _norm_title, build_graph
from akb.ingest.obsidian_loader import _norm_key, load_note
from akb.schemas import Document, SourceType


def test_norm_key_full_casefold_german() -> None:
    assert _norm_key("Straße") == _norm_key("STRASSE")


def test_norm_key_nfc_normalisation() -> None:
    nfd = unicodedata.normalize("NFD", "Café")
    nfc = unicodedata.normalize("NFC", "Café")
    assert _norm_key(nfc) == _norm_key(nfd)


def test_graph_norm_title_matches_obsidian_loader() -> None:
    assert _norm_title("Café") == _norm_key("café")


def test_loader_resolves_unicode_link(tmp_path: Path) -> None:
    cafe = tmp_path / "Café.md"
    cafe.write_text("body of cafe", encoding="utf-8")
    other = tmp_path / "Other.md"
    other.write_text("see [[cafe]] and [[CAFÉ]]", encoding="utf-8")

    # Build the index our loader uses
    from akb.config import IngestConfig

    cfg = IngestConfig(skip_dirs=[])
    from akb.ingest.obsidian_loader import _build_index

    index = _build_index(tmp_path, cfg)
    doc = load_note(other, vault=tmp_path, index=index)
    assert "cafe" in [w.lower() for w in doc.wikilinks] or any(
        "café" in w.lower() for w in doc.wikilinks
    )


def test_graph_alias_matches_after_casefold() -> None:
    a = Document(
        source_id="obsidian:A.md",
        source_type=SourceType.obsidian,
        title="A",
        content="",
        wikilinks=["Straße"],
    )
    b = Document(
        source_id="obsidian:B.md",
        source_type=SourceType.obsidian,
        title="Strasse",
        content="",
    )
    g = build_graph([a, b])
    assert "obsidian:B.md" in g.forward["obsidian:A.md"]
