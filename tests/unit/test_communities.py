"""Community detection on the wikilink graph + summary chunk shape."""

from __future__ import annotations

from akb.config import CommunitiesConfig
from akb.ingest.communities import _louvain, _make_chunk
from akb.ingest.graph import VaultGraph
from akb.schemas import SourceType


def test_louvain_returns_communities() -> None:
    g = VaultGraph(
        forward={
            "obsidian:A.md": {"obsidian:B.md"},
            "obsidian:B.md": {"obsidian:A.md", "obsidian:C.md"},
            "obsidian:C.md": {"obsidian:B.md"},
            "obsidian:X.md": {"obsidian:Y.md"},
            "obsidian:Y.md": {"obsidian:X.md", "obsidian:Z.md"},
            "obsidian:Z.md": {"obsidian:Y.md"},
            "obsidian:singleton.md": set(),
        },
        backward={},
    )
    found = _louvain(g, CommunitiesConfig(min_community_size=3))
    assert len(found) >= 1
    for members in found.values():
        assert len(members) >= 3


def test_louvain_filters_below_threshold() -> None:
    g = VaultGraph(
        forward={
            "obsidian:A.md": {"obsidian:B.md"},
            "obsidian:B.md": {"obsidian:A.md"},
        },
        backward={},
    )
    found = _louvain(g, CommunitiesConfig(min_community_size=5))
    assert found == {}


def test_make_chunk_shape() -> None:
    c = _make_chunk(0, ["obsidian:A.md", "obsidian:B.md"], "synthesis text")
    assert c.source_type == SourceType.community
    assert c.metadata["member_sources"] == ["obsidian:A.md", "obsidian:B.md"]
    assert "Community" in (c.metadata["title"] or "")
