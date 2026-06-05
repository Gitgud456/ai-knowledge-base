"""Integration test for the ingest pipeline: a temp Obsidian-shaped vault
→ documents → chunks → graph. Uses real loaders + chunker; no embedder.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from akb.config import IngestConfig, load_settings, reset_settings_cache
from akb.ingest.graph import build_graph
from akb.ingest.obsidian_loader import iter_vault, load_vault
from akb.ingest.pipeline import chunks_for_path


@pytest.fixture
def fake_vault(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Build a tiny Obsidian-shaped vault with frontmatter, tags, wikilinks, embeds."""
    (tmp_path / ".obsidian").mkdir()
    (tmp_path / ".obsidian" / "workspace").write_text("ignore me", encoding="utf-8")

    (tmp_path / "Index.md").write_text(
        "---\n"
        "tags: [moc, security]\n"
        "aliases: [\"MOC\"]\n"
        "---\n"
        "# Security MOC\n\n"
        "See [[ARP Spoofing]] and [[Python Decorators|decorators]].\n"
        "Also #network and ![[Snippet]]\n",
        encoding="utf-8",
    )
    (tmp_path / "ARP Spoofing.md").write_text(
        "---\ntags: [security, network]\n---\n"
        "# ARP Spoofing\n\nIntercepts traffic via crafted ARP responses.\n"
        "Refers back to [[Index]].\n",
        encoding="utf-8",
    )
    (tmp_path / "Python Decorators.md").write_text(
        "# Python Decorators\n\nWrap functions with @syntax.\n",
        encoding="utf-8",
    )
    (tmp_path / "Snippet.md").write_text("This is an embeddable snippet.\n", encoding="utf-8")

    # Point the loader at this vault via env
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(tmp_path))
    monkeypatch.setenv("AKB_PATHS__VAULT", str(tmp_path))
    reset_settings_cache()
    return tmp_path


def test_iter_vault_skips_dotobsidian(fake_vault: Path) -> None:
    docs = list(iter_vault(fake_vault))
    titles = sorted(d.title or "" for d in docs)
    assert titles == ["ARP Spoofing", "Index", "Python Decorators", "Snippet"]


def test_loader_extracts_frontmatter_tags_wikilinks(fake_vault: Path) -> None:
    docs = load_vault(fake_vault)
    by_title = {d.title: d for d in docs}
    index = by_title["Index"]
    assert "security" in index.tags
    assert "network" in index.tags  # inline #network
    assert "moc" in index.tags
    assert "MOC" in index.aliases
    assert set(index.wikilinks) >= {"ARP Spoofing", "Python Decorators", "Snippet"}


def test_embed_inlines_target(fake_vault: Path) -> None:
    docs = load_vault(fake_vault)
    index_doc = next(d for d in docs if d.title == "Index")
    assert "embeddable snippet" in index_doc.content


def test_wikilink_graph_round_trip(fake_vault: Path) -> None:
    docs = load_vault(fake_vault)
    g = build_graph(docs)
    sid_index = next(d.source_id for d in docs if d.title == "Index")
    sid_arp = next(d.source_id for d in docs if d.title == "ARP Spoofing")
    assert sid_arp in g.forward[sid_index]
    assert sid_index in g.backward[sid_arp]
    # Aliased link: "Python Decorators|decorators" → resolves to the doc
    sid_pd = next(d.source_id for d in docs if d.title == "Python Decorators")
    assert sid_pd in g.forward[sid_index]


def test_chunks_for_path_no_contextualize(fake_vault: Path) -> None:
    # contextualize=False keeps this test offline (no Ollama call).
    chunks = chunks_for_path(fake_vault, contextualize=False)
    assert chunks, "expected chunks"
    sources = {c.source_id for c in chunks}
    assert any("Index" in s for s in sources)
    assert any("ARP" in s for s in sources)
    # header_path must be populated for header-bearing notes
    arp = [c for c in chunks if "ARP" in c.source_id]
    assert any(c.header_path for c in arp), "expected header_path on ARP chunks"
