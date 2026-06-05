from __future__ import annotations

from akb.ingest.graph import build_graph
from akb.schemas import Document, SourceType


def _doc(sid: str, title: str, links: list[str], aliases: list[str] | None = None) -> Document:
    return Document(
        source_id=sid,
        source_type=SourceType.obsidian,
        title=title,
        content="",
        wikilinks=links,
        aliases=aliases or [],
    )


def test_forward_and_backward() -> None:
    a = _doc("obsidian:A.md", "A", ["B"])
    b = _doc("obsidian:B.md", "B", ["A", "C"])
    c = _doc("obsidian:C.md", "C", [])
    g = build_graph([a, b, c])
    assert g.forward["obsidian:A.md"] == {"obsidian:B.md"}
    assert g.forward["obsidian:B.md"] == {"obsidian:A.md", "obsidian:C.md"}
    assert g.backward["obsidian:A.md"] == {"obsidian:B.md"}
    assert g.backward["obsidian:C.md"] == {"obsidian:B.md"}


def test_alias_resolution() -> None:
    a = _doc("obsidian:A.md", "A", ["Beta"])
    b = _doc("obsidian:B.md", "B", [], aliases=["Beta"])
    g = build_graph([a, b])
    assert g.forward["obsidian:A.md"] == {"obsidian:B.md"}


def test_neighbours_multi_hop() -> None:
    a = _doc("obsidian:A.md", "A", ["B"])
    b = _doc("obsidian:B.md", "B", ["C"])
    c = _doc("obsidian:C.md", "C", [])
    g = build_graph([a, b, c])
    one_hop = g.neighbours("obsidian:A.md", hops=1)
    two_hop = g.neighbours("obsidian:A.md", hops=2)
    assert one_hop == {"obsidian:B.md"}
    assert two_hop == {"obsidian:B.md", "obsidian:C.md"}
