"""Contextual Retrieval (Anthropic, Sept 2024) — local-LLM variant.

For each chunk, ask the local Ollama model to write a 50-100 token "situating"
prefix that explains where the chunk lives in its parent document. We then embed
*context + chunk* instead of the bare chunk. Reported ~35% reduction in retrieval
failure for the embedding side, ~49% with BM25, ~67% combined with a reranker.

Design choices:
  * **Batched LLM calls.** Instead of one ``ollama.generate`` per chunk, we
    bundle up to ``ingest.context_batch_size`` chunks into a single prompt that
    returns a JSON array. Llama 3 8B handles this comfortably at 16/batch and
    cuts wall time ~8-12x for cold ingest of a large vault.
  * **Per-row autocommit cache.** A Ctrl-C halfway through a long document does
    NOT lose contexts the model already produced — the cache row was committed
    immediately.
  * **Fail-open.** If the LLM call errors or the JSON is malformed, the
    affected chunks go through with their original text. We never block
    ingestion on the contextualizer.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import ollama

from akb.config import IngestConfig, LLMConfig, load_settings
from akb.obs.logging import get_logger
from akb.schemas import Chunk, Document

log = get_logger(__name__)

_SINGLE_PROMPT = """<document>
{document}
</document>

Here is a chunk we want to situate within the whole document:

<chunk>
{chunk}
</chunk>

Please give a short, succinct context (1-2 sentences, max 100 tokens) to situate this chunk within the overall document, for the purposes of improving search retrieval of the chunk. Answer with ONLY the context, nothing else."""

_BATCH_PROMPT = """<document>
{document}
</document>

Here are several chunks from the document. For each, write a 1-2 sentence context (≤100 tokens) that situates the chunk within the overall document, for the purposes of improving search retrieval.

Respond with ONLY a JSON object of the form:
{{"contexts": [{{"id": <int>, "context": "..."}}, ...]}}

Chunks:
{chunks}
"""


@contextmanager
def _cache_conn(path: Path) -> Iterator[sqlite3.Connection]:
    path.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(path, isolation_level=None)  # autocommit; callers persist per row
    try:
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA synchronous=NORMAL")
        c.execute(
            "CREATE TABLE IF NOT EXISTS context_cache ("
            "key TEXT PRIMARY KEY, "
            "context TEXT NOT NULL, "
            "created_at TEXT NOT NULL)"
        )
        yield c
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


def _trim_document(document_text: str, max_chars: int = 12000) -> str:
    if len(document_text) <= max_chars:
        return document_text
    half = max_chars // 2
    return document_text[:half] + "\n\n[...]\n\n" + document_text[-half:]


def _format_batch(chunks: list[tuple[int, str]]) -> str:
    parts: list[str] = []
    for i, text in chunks:
        parts.append(f'<chunk id="{i}">\n{text}\n</chunk>')
    return "\n".join(parts)


_JSON_OBJ_RX = re.compile(r"\{.*\}", re.DOTALL)


def _parse_batch_response(raw: str) -> dict[int, str]:
    """Parse the LLM's JSON-mode response. Tolerant of leading/trailing text."""
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = _JSON_OBJ_RX.search(raw)
        if not m:
            return {}
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return {}
    out: dict[int, str] = {}
    for row in data.get("contexts", []):
        if not isinstance(row, dict):
            continue
        try:
            idx = int(row.get("id"))
            ctx = str(row.get("context", "")).strip()
        except (TypeError, ValueError):
            continue
        if ctx:
            out[idx] = ctx
    return out


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

    # --- single-chunk fallback (kept for resilience + tests) ---

    def _generate_one(self, document_text: str, chunk_text: str) -> str:
        prompt = _SINGLE_PROMPT.format(document=_trim_document(document_text), chunk=chunk_text)
        try:
            resp = ollama.generate(
                model=self._model,
                prompt=prompt,
                options={"temperature": 0.0, "num_predict": 160},
            )
            return str(resp.get("response", "")).strip()
        except Exception as e:
            log.warning("contextualize.generate.error", error=str(e))
            return ""

    # --- batched path ---

    def _generate_batch(
        self,
        document_text: str,
        items: list[tuple[int, str]],
    ) -> dict[int, str]:
        if not items:
            return {}
        prompt = _BATCH_PROMPT.format(
            document=_trim_document(document_text),
            chunks=_format_batch(items),
        )
        try:
            resp = ollama.generate(
                model=self._model,
                prompt=prompt,
                format="json",
                options={"temperature": 0.0, "num_predict": 160 * len(items) + 128},
            )
            return _parse_batch_response(str(resp.get("response", "")))
        except Exception as e:
            log.warning("contextualize.batch.error", error=str(e), batch=len(items))
            return {}

    # --- main entry ---

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
        batch_size = max(1, self._ingest.context_batch_size)
        hits = 0
        misses = 0

        with _cache_conn(self._cache_path) as conn:
            # First: serve everything we can from cache, list misses by index.
            todo: list[tuple[int, str, str]] = []  # (chunk_index_in_list, cache_key, text)
            for i, chunk in enumerate(chunks):
                key = _cache_key(document.source_id, doc_hash, chunk.text)
                row = conn.execute(
                    "SELECT context FROM context_cache WHERE key = ?", (key,)
                ).fetchone()
                if row is not None:
                    if row[0]:
                        chunk.contextualized_text = f"{row[0]}\n\n{chunk.text}"
                    hits += 1
                else:
                    todo.append((i, key, chunk.text))

            # Then: batched generation for the misses.
            for start in range(0, len(todo), batch_size):
                slice_ = todo[start : start + batch_size]
                items = [(i, text) for i, _key, text in slice_]
                generated = self._generate_batch(document_text, items)

                # If the batch failed wholesale, fall back to per-chunk generation
                # so a malformed JSON doesn't lose the whole batch.
                if not generated and len(slice_) > 1:
                    for i, _key, text in slice_:
                        single = self._generate_one(document_text, text)
                        if single:
                            generated[i] = single

                for i, key, text in slice_:
                    ctx = generated.get(i, "")
                    misses += 1
                    if ctx:
                        conn.execute(
                            "INSERT OR REPLACE INTO context_cache (key, context, created_at) VALUES (?, ?, ?)",
                            (key, ctx, datetime.now().isoformat()),
                        )
                        chunks[i].contextualized_text = f"{ctx}\n\n{text}"

        log.info(
            "contextualize.done",
            source_id=document.source_id,
            hits=hits,
            misses=misses,
            batch_size=batch_size,
        )
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
