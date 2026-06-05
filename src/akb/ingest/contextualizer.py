"""Contextual Retrieval (Anthropic, Sept 2024) — local-LLM variant.

For each chunk, ask the local Ollama model to write a 50-100 token "situating"
prefix that explains where the chunk lives in its parent document. We then embed
*context + chunk* instead of the bare chunk. Reported ~35% reduction in retrieval
failure for the embedding side, ~49% with BM25, ~67% combined with a reranker.

Design choices:
  * One LLM call per chunk. We deliberately do NOT batch the whole doc — that
    blows the context window. We *do* pass the full doc as context per chunk
    so the model sees the whole picture each time. Token cost stays manageable
    because we run a local 8B model, not an API.
  * We cache by ``(source_id, content_hash, chunk_text_hash)`` so re-ingestion
    of a touched note only re-contextualizes its *changed* chunks.
  * Fail-open: if the LLM call errors, the chunk goes through with its original
    text. We never block ingestion on the contextualizer.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterable
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

import ollama

from akb.config import IngestConfig, LLMConfig, load_settings
from akb.schemas import Chunk, Document

_PROMPT = """<document>
{document}
</document>

Here is a chunk we want to situate within the whole document:

<chunk>
{chunk}
</chunk>

Please give a short, succinct context (1-2 sentences, max 100 tokens) to situate this chunk within the overall document, for the purposes of improving search retrieval of the chunk. Answer with ONLY the context, nothing else."""


@contextmanager
def _cache_conn(path: Path) -> Iterator[sqlite3.Connection]:
    path.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(path)
    try:
        c.execute(
            "CREATE TABLE IF NOT EXISTS context_cache ("
            "key TEXT PRIMARY KEY, "
            "context TEXT NOT NULL, "
            "created_at TEXT NOT NULL)"
        )
        yield c
        c.commit()
    finally:
        c.close()


def _cache_key(source_id: str, doc_hash: str, chunk_text: str) -> str:
    h = hashlib.sha256()
    h.update(source_id.encode())
    h.update(b"|")
    h.update(doc_hash.encode())
    h.update(b"|")
    h.update(chunk_text.encode())
    return h.hexdigest()


class Contextualizer:
    """Generates and caches situating prefixes for chunks."""

    def __init__(
        self,
        cache_path: Path | None = None,
        llm_cfg: LLMConfig | None = None,
        ingest_cfg: IngestConfig | None = None,
    ) -> None:
        settings = load_settings()
        self._cache_path = cache_path or (settings.paths.data_dir / "context_cache.db")
        self._llm = llm_cfg or settings.llm
        self._ingest = ingest_cfg or settings.ingest
        self._model = self._llm.context_model

    def _generate(self, document_text: str, chunk_text: str) -> str:
        # Trim document to a sane window so the local model doesn't OOM on long docs.
        # 12k chars ≈ 3k tokens; the chunk itself adds ~300-500.
        max_chars = 12000
        doc = (
            document_text
            if len(document_text) <= max_chars
            else (document_text[: max_chars // 2] + "\n\n[...]\n\n" + document_text[-max_chars // 2 :])
        )
        prompt = _PROMPT.format(document=doc, chunk=chunk_text)
        try:
            resp = ollama.generate(
                model=self._model,
                prompt=prompt,
                options={"temperature": 0.0, "num_predict": 160},
            )
            return str(resp.get("response", "")).strip()
        except Exception:
            return ""

    def contextualize(
        self,
        document: Document,
        chunks: list[Chunk],
    ) -> list[Chunk]:
        """Mutates chunks to set ``contextualized_text = context + '\\n\\n' + text``."""
        if not chunks or not self._ingest.contextual_retrieval:
            return chunks

        doc_hash = document.content_hash
        document_text = document.content

        with _cache_conn(self._cache_path) as conn:
            for chunk in chunks:
                key = _cache_key(document.source_id, doc_hash, chunk.text)
                row = conn.execute(
                    "SELECT context FROM context_cache WHERE key = ?", (key,)
                ).fetchone()
                if row is not None:
                    context = row[0]
                else:
                    context = self._generate(document_text, chunk.text)
                    if context:
                        conn.execute(
                            "INSERT OR REPLACE INTO context_cache (key, context, created_at) VALUES (?, ?, ?)",
                            (key, context, datetime.now().isoformat()),
                        )
                if context:
                    chunk.contextualized_text = f"{context}\n\n{chunk.text}"
        return chunks

    def stats(self) -> dict[str, int]:
        with _cache_conn(self._cache_path) as conn:
            (n,) = conn.execute("SELECT COUNT(*) FROM context_cache").fetchone()
        return {"cached": int(n)}


def contextualize_pairs(
    pairs: Iterable[tuple[Document, list[Chunk]]],
    contextualizer: Contextualizer | None = None,
) -> Iterator[tuple[Document, list[Chunk]]]:
    """Stream-friendly wrapper used by the ingest pipeline."""
    ctx = contextualizer or Contextualizer()
    for doc, chunks in pairs:
        yield doc, ctx.contextualize(doc, chunks)


_ = json  # reserved for future structured-output mode
