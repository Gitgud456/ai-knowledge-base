"""Chunker invariants:
  * header_path is preserved per section
  * no chunk is empty
  * chunk_index is monotonic per document
  * oversize sections get sub-split
"""

from __future__ import annotations

from akb.config import IngestConfig
from akb.ingest.chunkers import chunk_document
from akb.schemas import Document, SourceType


def _doc(text: str) -> Document:
    return Document(source_id="t:doc1", source_type=SourceType.txt, content=text)


def test_no_headers_falls_back_to_recursive() -> None:
    text = "para one.\n\npara two.\n\npara three."
    chunks = chunk_document(_doc(text), IngestConfig(chunk_size=50, chunk_overlap=10))
    assert chunks, "expected at least one chunk"
    assert all(c.text.strip() for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_header_path_preserved() -> None:
    text = (
        "# H1\n\n"
        "intro under h1\n\n"
        "## H2 A\n\n"
        "h2a body\n\n"
        "## H2 B\n\n"
        "h2b body\n"
    )
    chunks = chunk_document(_doc(text), IngestConfig(chunk_size=500, chunk_overlap=50))
    paths = [tuple(c.header_path) for c in chunks]
    assert any("H1" in p[0] for p in paths if p), "H1 missing from header_path"
    assert any(len(p) >= 2 and "H2 A" in p[1] for p in paths), "H2 not nested under H1"


def test_long_section_is_subsplit() -> None:
    long_para = " ".join(["word"] * 1000)
    text = f"# H1\n\n{long_para}"
    chunks = chunk_document(_doc(text), IngestConfig(chunk_size=200, chunk_overlap=20))
    assert len(chunks) > 1
    for c in chunks:
        assert c.header_path == ["H1"]
        assert len(c.text) <= 400  # generous; respect overlap + boundary slack
