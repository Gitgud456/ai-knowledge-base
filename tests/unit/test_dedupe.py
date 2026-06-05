from __future__ import annotations

from akb.ingest.dedupe import dedupe_chunks, sha256_norm
from akb.schemas import Chunk, SourceType


def _c(text: str, i: int = 0) -> Chunk:
    return Chunk(source_id="t", source_type=SourceType.txt, text=text, chunk_index=i)


def test_sha256_norm_collapses_whitespace_and_case() -> None:
    assert sha256_norm("Hello\n\nWorld") == sha256_norm("hello world")


def test_dedupe_exact() -> None:
    out = dedupe_chunks([_c("hello world", 0), _c("hello\n\nworld", 1)])
    assert len(out) == 1


def test_dedupe_near() -> None:
    a = "the quick brown fox jumps over the lazy dog and the moon was full"
    b = "the quick brown fox jumps over the lazy dog and the moon was bright"
    out = dedupe_chunks([_c(a, 0), _c(b, 1)], near_dup_threshold=0.7)
    assert len(out) == 1


def test_dedupe_keeps_distinct() -> None:
    out = dedupe_chunks([_c("alpha bravo", 0), _c("zulu yankee", 1)])
    assert len(out) == 2
