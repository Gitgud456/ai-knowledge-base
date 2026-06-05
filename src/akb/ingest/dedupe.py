"""Chunk-level deduplication.

Two cheap layers:
  * exact SHA-256 of normalized text (whitespace-collapsed, lowercased)
  * length-bucketed shingle hashing for near-duplicates (template boilerplate
    in Obsidian vaults is the common case)

This runs before upsert; downstream the vector store sees fewer rows.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

from akb.schemas import Chunk

_WS_RX = re.compile(r"\s+")


def _normalise(text: str) -> str:
    return _WS_RX.sub(" ", text).strip().lower()


def sha256_norm(text: str) -> str:
    return hashlib.sha256(_normalise(text).encode("utf-8")).hexdigest()


def _shingles(text: str, k: int = 7) -> set[str]:
    words = _normalise(text).split()
    if len(words) < k:
        return {" ".join(words)}
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def dedupe_chunks(chunks: Iterable[Chunk], near_dup_threshold: float = 0.92) -> list[Chunk]:
    """Drop exact duplicates and near-duplicates within the input batch.

    Cross-batch dedup is the ingest pipeline's job (it checks ``ingest_state.db``).
    """
    out: list[Chunk] = []
    seen_hashes: set[str] = set()
    shingle_index: list[set[str]] = []

    for c in chunks:
        h = sha256_norm(c.text)
        if h in seen_hashes:
            continue
        s = _shingles(c.text)
        if any(jaccard(s, prev) >= near_dup_threshold for prev in shingle_index):
            continue
        seen_hashes.add(h)
        shingle_index.append(s)
        out.append(c)
    return out
