"""EPUB loader (ebooklib + BeautifulSoup)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from akb.schemas import Document, SourceType


def _extract(path: Path) -> str:
    import ebooklib
    from bs4 import BeautifulSoup
    from ebooklib import epub

    book = epub.read_epub(str(path))
    parts: list[str] = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "html.parser")
        parts.append(soup.get_text("\n", strip=True))
    return "\n\n".join(parts)


def load_epub(path: Path) -> Document:
    text = _extract(path)
    stat = path.stat()
    return Document(
        source_id=f"epub:{path.name}",
        source_path=path,
        source_type=SourceType.epub,
        title=path.stem,
        content=text,
        created_at=datetime.fromtimestamp(stat.st_ctime),
        modified_at=datetime.fromtimestamp(stat.st_mtime),
        extra={"size": stat.st_size},
    )
