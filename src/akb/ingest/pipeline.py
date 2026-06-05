"""Top-level ingest orchestration.

Path → Document(s) → Chunk(s) → dedupe → (Phase 4: contextualize) → ready for upsert.

This module is pure orchestration — no DB writes here. The store layer (Phase 2)
will own upsert; the sync layer (Phase 6) will own change detection.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

from akb.config import load_settings
from akb.ingest.chunkers import chunk_document
from akb.ingest.contextualizer import Contextualizer
from akb.ingest.dedupe import dedupe_chunks
from akb.ingest.epub_loader import load_epub
from akb.ingest.obsidian_loader import iter_vault, load_note
from akb.ingest.pdf_loader import load_pdf
from akb.ingest.txt_loader import load_txt
from akb.schemas import Chunk, Document, SourceType


def load_single(path: Path) -> Document:
    """Dispatch by extension for a single non-vault file."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    if suffix == ".epub":
        return load_epub(path)
    if suffix == ".txt":
        return load_txt(path)
    if suffix == ".md":
        # Treat lone .md as a 1-file mini-vault.
        return load_note(path, vault=path.parent, index={path.stem.lower(): path})
    raise ValueError(f"Unsupported extension: {suffix}")


def iter_documents(path: Path | None = None) -> Iterator[Document]:
    """Yield Documents for either a single file, a directory, or (default) the configured vault."""
    if path is None:
        yield from iter_vault()
        return
    if path.is_file():
        yield load_single(path)
        return
    if path.is_dir():
        # If it looks like the configured vault, use the full vault loader (handles embeds).
        cfg_vault = load_settings().paths.vault
        if path.resolve() == cfg_vault.resolve():
            yield from iter_vault(path)
            return
        # Otherwise: walk + dispatch per extension.
        for f in sorted(path.rglob("*")):
            if f.is_file() and f.suffix.lower() in {".pdf", ".epub", ".txt", ".md"}:
                try:
                    yield load_single(f)
                except Exception:
                    continue
        return
    raise FileNotFoundError(path)


def chunks_for(
    documents: Iterable[Document],
    *,
    contextualize: bool | None = None,
) -> list[Chunk]:
    """Chunk + (optionally contextualize) + dedupe a stream of documents.

    ``contextualize=None`` honors the config flag (Phase 4 default: True).
    """
    settings = load_settings()
    if contextualize is None:
        contextualize = settings.ingest.contextual_retrieval

    ctx = Contextualizer() if contextualize else None
    all_chunks: list[Chunk] = []
    for doc in documents:
        chunks = chunk_document(doc)
        if ctx is not None:
            chunks = ctx.contextualize(doc, chunks)
        all_chunks.extend(chunks)
    return dedupe_chunks(all_chunks)


def chunks_for_path(
    path: Path | None = None,
    *,
    contextualize: bool | None = None,
) -> list[Chunk]:
    """Convenience: path → chunks (chunked + optionally contextualized + deduped)."""
    return chunks_for(iter_documents(path), contextualize=contextualize)


def summarize_documents(docs: Iterable[Document]) -> dict[str, int]:
    """Quick stats for the CLI ``akb info`` style output."""
    by_type: dict[str, int] = {}
    for d in docs:
        by_type[d.source_type.value] = by_type.get(d.source_type.value, 0) + 1
    return by_type


_ = SourceType  # re-exported convenience for callers
