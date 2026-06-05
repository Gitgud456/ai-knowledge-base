"""PDF loader.

Prefers ``pymupdf4llm`` (extracts markdown with real headers — gold for chunking).
Falls back to raw ``pymupdf`` text if pymupdf4llm fails for any reason.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from akb.schemas import Document, SourceType


def _extract_markdown(path: Path) -> str:
    try:
        import pymupdf4llm

        return pymupdf4llm.to_markdown(str(path))
    except Exception:
        import pymupdf  # type: ignore

        doc = pymupdf.open(str(path))
        try:
            return "\n\n".join(page.get_text() for page in doc)  # type: ignore[no-untyped-call]
        finally:
            doc.close()


def load_pdf(path: Path) -> Document:
    text = _extract_markdown(path)
    stat = path.stat()
    return Document(
        source_id=f"pdf:{path.name}",
        source_path=path,
        source_type=SourceType.pdf,
        title=path.stem,
        content=text,
        created_at=datetime.fromtimestamp(stat.st_ctime),
        modified_at=datetime.fromtimestamp(stat.st_mtime),
        extra={"size": stat.st_size},
    )
