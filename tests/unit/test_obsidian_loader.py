"""Unit tests for the Obsidian loader — parsing rules that have to be right
(wikilinks, embed expansion, tag extraction, frontmatter handling)."""

from __future__ import annotations

from pathlib import Path

import pytest

from akb.ingest.obsidian_loader import (
    _extract_inline_tags,
    _extract_wikilinks,
    _expand_embeds,
    _split_link,
    load_note,
)


def test_split_link_plain() -> None:
    assert _split_link("Some Note") == ("Some Note", None, None)


def test_split_link_alias() -> None:
    assert _split_link("Note|Alias") == ("Note", None, "Alias")


def test_split_link_heading() -> None:
    assert _split_link("Note#Section") == ("Note", "Section", None)


def test_split_link_alias_and_heading() -> None:
    assert _split_link("Note#Section|Alias") == ("Note", "Section", "Alias")


def test_extract_wikilinks_dedup_preserves_order() -> None:
    body = "see [[A]] and [[B|alias]] and [[A]] and [[C#h]]"
    assert _extract_wikilinks(body) == ["A", "B", "C"]


def test_extract_inline_tags() -> None:
    body = "Topic #security and #ml/intro plus mid-sentence#nope-not-a-tag and #ai"
    tags = set(_extract_inline_tags(body))
    assert {"security", "ml/intro", "ai"} <= tags
    # word-internal # should not match
    assert "nope-not-a-tag" not in tags


def test_expand_embeds(tmp_path: Path) -> None:
    a = tmp_path / "A.md"
    b = tmp_path / "B.md"
    a.write_text("Body A with ![[B]]", encoding="utf-8")
    b.write_text("Body B content", encoding="utf-8")
    index = {"a": a, "b": b}
    out = _expand_embeds("![[A]]", base=tmp_path, vault=tmp_path, index=index, seen=set())
    assert "Body A" in out
    assert "Body B" in out


def test_expand_embeds_cycle_safe(tmp_path: Path) -> None:
    a = tmp_path / "A.md"
    b = tmp_path / "B.md"
    a.write_text("![[B]]", encoding="utf-8")
    b.write_text("![[A]]", encoding="utf-8")
    index = {"a": a, "b": b}
    # Should not infinite-loop; depth is bounded.
    out = _expand_embeds("![[A]]", base=tmp_path, vault=tmp_path, index=index, seen=set())
    assert isinstance(out, str)


def test_load_note_round_trip(tmp_path: Path) -> None:
    p = tmp_path / "MyNote.md"
    p.write_text(
        "---\n"
        "tags: [project, ai]\n"
        "aliases: [\"MN\", \"Note\"]\n"
        "---\n"
        "# Heading\n\n"
        "Body refers to [[Other]] and uses #inline.\n",
        encoding="utf-8",
    )
    doc = load_note(p, vault=tmp_path, index={"mynote": p})
    assert doc.title == "MyNote"
    assert "project" in doc.tags
    assert "ai" in doc.tags
    assert "inline" in doc.tags
    assert "MN" in doc.aliases
    assert "Other" in doc.wikilinks


@pytest.mark.parametrize("bad_yaml", ["---\n: invalid : yaml\n---\nBody", "---\n---\nBody"])
def test_load_note_resilient_to_bad_frontmatter(tmp_path: Path, bad_yaml: str) -> None:
    p = tmp_path / "Broken.md"
    p.write_text(bad_yaml, encoding="utf-8")
    doc = load_note(p, vault=tmp_path, index={"broken": p})
    assert "Body" in doc.content
