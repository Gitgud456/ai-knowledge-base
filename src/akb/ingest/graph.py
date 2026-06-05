"""Wikilink graph builder.

Walks the vault (or any iterable of Documents) and produces:
  * ``forward_links``: source_id -> set[source_id]   (outbound wikilinks)
  * ``backlinks``:     source_id -> set[source_id]   (inbound wikilinks)
  * A NetworkX DiGraph for arbitrary traversal / visualisation.

Used by retrieve.graph_expand (Phase 2+) to do 1-hop expansion at query time
and by the new Knowledge Graph tab to render the *real* graph instead of an
embedding-similarity hairball.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import networkx as nx

from akb.schemas import Document


@dataclass(frozen=False)
class VaultGraph:
    forward: dict[str, set[str]] = field(default_factory=dict)
    backward: dict[str, set[str]] = field(default_factory=dict)
    title_to_source: dict[str, str] = field(default_factory=dict)

    def neighbours(self, source_id: str, hops: int = 1) -> set[str]:
        out: set[str] = set()
        frontier = {source_id}
        for _ in range(max(hops, 0)):
            nxt: set[str] = set()
            for s in frontier:
                nxt |= self.forward.get(s, set())
                nxt |= self.backward.get(s, set())
            nxt -= out
            nxt.discard(source_id)
            out |= nxt
            frontier = nxt
            if not frontier:
                break
        return out

    def to_networkx(self) -> nx.DiGraph:
        g: nx.DiGraph = nx.DiGraph()
        for src, targets in self.forward.items():
            for t in targets:
                g.add_edge(src, t)
        return g


def _norm_title(s: str) -> str:
    return s.strip().lower()


def build_graph(documents: Iterable[Document]) -> VaultGraph:
    """Resolve wikilinks across the document set into a graph keyed by source_id."""
    docs = list(documents)
    title_to_source: dict[str, str] = {}
    for d in docs:
        if d.title:
            title_to_source.setdefault(_norm_title(d.title), d.source_id)
        # also allow exact source_id self-match
        title_to_source.setdefault(_norm_title(d.source_id), d.source_id)
        for alias in d.aliases:
            title_to_source.setdefault(_norm_title(alias), d.source_id)

    forward: dict[str, set[str]] = {d.source_id: set() for d in docs}
    backward: dict[str, set[str]] = {d.source_id: set() for d in docs}

    for d in docs:
        for link in d.wikilinks:
            tgt = title_to_source.get(_norm_title(link))
            if not tgt or tgt == d.source_id:
                continue
            forward[d.source_id].add(tgt)
            backward.setdefault(tgt, set()).add(d.source_id)

    return VaultGraph(forward=forward, backward=backward, title_to_source=title_to_source)
