"""Chunking strategies.

The default pipeline is:
  1. Try a markdown-header-aware split (preserves H1>H2>H3 breadcrumb).
  2. For each header-section that's still too large, recursive-split it.
  3. Carry a `header_path` and `chunk_index` on every emitted Chunk.

This is structurally smarter than the legacy ``RecursiveCharacterTextSplitter`` at root,
and yields metadata the retriever can use (filter on H1 == "Project X", etc.).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from akb.config import IngestConfig, load_settings
from akb.schemas import Chunk, Document


@dataclass(frozen=True)
class _Section:
    text: str
    header_path: list[str]


def _to_md_splitter(cfg: IngestConfig) -> MarkdownHeaderTextSplitter:
    headers = [(h[0], h[1]) for h in cfg.headers_to_split]
    return MarkdownHeaderTextSplitter(headers_to_split_on=headers, strip_headers=False)


def _to_recursive(cfg: IngestConfig) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        # Code-aware: prefer paragraph > sentence > word boundaries.
        separators=["\n\n", "\n", "```", ". ", "? ", "! ", " ", ""],
    )


_PROTECTED_HEADER_RX = re.compile(r"^\s*#{1,6}\s+", re.MULTILINE)


def _has_markdown_headers(text: str) -> bool:
    return bool(_PROTECTED_HEADER_RX.search(text))


def _split_with_headers(text: str, cfg: IngestConfig) -> list[_Section]:
    if not _has_markdown_headers(text):
        return [_Section(text=text, header_path=[])]
    md = _to_md_splitter(cfg)
    docs = md.split_text(text)
    out: list[_Section] = []
    for d in docs:
        # d.metadata = {"h1": "...", "h2": "...", ...} (only keys present)
        path = [d.metadata[k] for k in ("h1", "h2", "h3", "h4") if d.metadata.get(k)]
        out.append(_Section(text=d.page_content, header_path=path))
    return out


def _recursive_split(text: str, cfg: IngestConfig) -> list[str]:
    return _to_recursive(cfg).split_text(text)


def chunk_document(
    doc: Document,
    cfg: IngestConfig | None = None,
) -> list[Chunk]:
    """Chunk one Document into a list[Chunk], header-aware.

    Sections smaller than ``chunk_size`` go through whole. Larger ones are
    sub-split by ``RecursiveCharacterTextSplitter`` while keeping the header
    breadcrumb on each piece.
    """
    cfg = cfg or load_settings().ingest
    sections = _split_with_headers(doc.content, cfg)
    chunks: list[Chunk] = []

    index = 0
    for section in sections:
        pieces: Iterable[str]
        if len(section.text) <= cfg.chunk_size:
            pieces = [section.text]
        else:
            pieces = _recursive_split(section.text, cfg)
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            chunks.append(
                Chunk(
                    source_id=doc.source_id,
                    source_type=doc.source_type,
                    text=piece,
                    header_path=section.header_path,
                    chunk_index=index,
                    tags=list(doc.tags),
                    wikilinks=list(doc.wikilinks),
                    metadata={
                        "title": doc.title,
                        "aliases": list(doc.aliases),
                        "frontmatter_keys": list(doc.frontmatter.keys()),
                        # Timestamps for recency-weighted rerank. ISO-8601 strings
                        # because Qdrant payload doesn't take datetime objects.
                        "created_at": doc.created_at.isoformat() if doc.created_at else None,
                        "modified_at": doc.modified_at.isoformat() if doc.modified_at else None,
                    },
                )
            )
            index += 1
    return chunks
