"""Sticky context via ``[[Note Name]]`` mentions in chat input.

When the user types ``[[Project Plan]] what's left?`` we want the agent to
treat *Project Plan* as the authoritative source — not hope retrieval picks
it. This module parses wikilinks from chat input and pulls the relevant
chunks straight from Qdrant by ``source_id`` filter, ready to be prepended to
the retrieved context as ``[pinned: title]`` blocks.

Resolution rules match the loader:
  * case-insensitive, Unicode-casefold
  * alias-aware via the VaultGraph if one is supplied
  * unresolved mentions are dropped (not an error — the user can verify in
    the reasoning panel)
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from akb.ingest.graph import VaultGraph
from akb.schemas import Chunk
from akb.store.qdrant_store import QdrantStore

_MENTION_RX = re.compile(r"\[\[([^\]\n]+?)\]\]")


def _norm(s: str) -> str:
    return unicodedata.normalize("NFC", s.strip()).casefold()


@dataclass(frozen=True)
class Pinned:
    title: str
    source_id: str
    chunks: list[Chunk]


def extract_mentions(text: str) -> list[str]:
    """Pull ``[[Note]]`` / ``[[Note|alias]]`` / ``[[Note#heading]]`` targets."""
    out: list[str] = []
    seen: set[str] = set()
    for m in _MENTION_RX.finditer(text):
        raw = m.group(1)
        target = raw
        if "|" in target:
            target = target.split("|", 1)[0]
        if "#" in target:
            target = target.split("#", 1)[0]
        t = target.strip()
        if t and _norm(t) not in seen:
            out.append(t)
            seen.add(_norm(t))
    return out


def strip_mentions(text: str) -> str:
    """Return the user's query with the ``[[...]]`` wrappers removed.

    Keeps the visible name so the LLM still sees the topic, but doesn't
    confuse the model with markdown link syntax.
    """
    def _replace(m: re.Match[str]) -> str:
        raw = m.group(1)
        if "|" in raw:
            _target, _, alias = raw.partition("|")
            return alias.strip()
        target = raw.split("#", 1)[0] if "#" in raw else raw
        return target.strip()

    return _MENTION_RX.sub(_replace, text)


def resolve(
    mentions: list[str],
    graph: VaultGraph | None,
    store: QdrantStore,
    *,
    limit_per_source: int = 6,
) -> list[Pinned]:
    """Look up source IDs and fetch their chunks. Unresolved mentions are dropped."""
    if not mentions:
        return []

    title_index: dict[str, str] = {}
    if graph is not None:
        title_index = {_norm(k): v for k, v in graph.title_to_source.items()}

    out: list[Pinned] = []
    seen_sids: set[str] = set()
    for raw_title in mentions:
        key = _norm(raw_title)
        sid = title_index.get(key)
        if not sid:
            # Fall back to obsidian:<title>.md guess (cheap, often correct)
            sid = f"obsidian:{raw_title}.md"
        if sid in seen_sids:
            continue
        chunks = store.fetch_chunks_for_sources([sid], limit_per_source=limit_per_source)
        if not chunks:
            continue
        seen_sids.add(sid)
        out.append(Pinned(title=raw_title, source_id=sid, chunks=chunks))
    return out


def format_pinned_block(pinned: list[Pinned]) -> str:
    """Render pinned chunks for prepending to the agent's CONTEXT."""
    if not pinned:
        return ""
    parts: list[str] = []
    for p in pinned:
        body = "\n\n".join(c.text for c in p.chunks)
        parts.append(f"[pinned: {p.title}]\n{body}")
    return "\n---\n".join(parts)
