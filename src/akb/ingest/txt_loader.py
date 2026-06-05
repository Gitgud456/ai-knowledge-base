"""Plain text loader."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from akb.schemas import Document, SourceType


def load_txt(path: Path) -> Document:
    text = path.read_text(encoding="utf-8", errors="replace")
    stat = path.stat()
    return Document(
        source_id=f"txt:{path.name}",
        source_path=path,
        source_type=SourceType.txt,
        title=path.stem,
        content=text,
        created_at=datetime.fromtimestamp(stat.st_ctime),
        modified_at=datetime.fromtimestamp(stat.st_mtime),
    )
