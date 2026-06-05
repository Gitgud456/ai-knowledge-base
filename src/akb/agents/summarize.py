"""Long-doc / multi-source summarization workflow.

Two entry points:

  * :func:`summarize_source` — every chunk of one source is fed through a map
    step (key points), then a reduce step (structured brief). Used for "read
    this 30-page PDF for me".
  * :func:`summarize_filter` — same, but the unit of work is everything
    matching a tag / source-type / arbitrary payload filter. Used for
    "summarize all my March notes" or "give me the gist of all #project notes".

Both return a :class:`SummaryResult` carrying the final brief, the per-chunk
notes (for inspection / drill-down in the UI), and the list of chunks the
summary cited so we can render them as Obsidian-friendly wikilinks.

Failure modes are quiet: if a map call fails for a chunk, that chunk is
skipped and the reduce step proceeds with what it has.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import ollama

from akb.config import LLMConfig, load_settings
from akb.obs.logging import get_logger
from akb.schemas import Chunk
from akb.store.qdrant_store import COLLECTION, QdrantStore, get_store

log = get_logger(__name__)


MAP_PROMPT = """You are summarising one chunk of a larger document.

Extract the 3-6 most important key points from this chunk. Be terse —
bullet points, no preamble, no recap of "what this chunk is about".

CHUNK:
{chunk}

KEY POINTS:"""


REDUCE_PROMPT = """You are merging key-point notes taken from many chunks of {scope}.

Produce a structured brief with these sections:
  * **Overview** — 2-3 sentences of what this material covers
  * **Key findings** — the most load-bearing points, bulleted
  * **Open questions** — anything the material flags as unresolved
  * **Next steps** — actionable if the material implies any

Be specific. Cite source chunks by their bracketed label when useful.

NOTES:
{notes}

STRUCTURED BRIEF:"""


@dataclass
class SummaryResult:
    text: str
    map_notes: list[str] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    scope: str = ""
    model: str = ""


def _label(chunk: Chunk) -> str:
    title = chunk.metadata.get("title") or chunk.source_id
    header = " > ".join(chunk.header_path) if chunk.header_path else ""
    return f"[{title}{(' :: ' + header) if header else ''}]"


def _map_one(chunk: Chunk, llm_cfg: LLMConfig) -> str:
    prompt = MAP_PROMPT.format(chunk=chunk.text)
    try:
        resp = ollama.generate(
            model=llm_cfg.local_model,
            prompt=prompt,
            options={"temperature": 0.0, "num_predict": 400},
        )
        body = str(resp.get("response", "")).strip()
        return f"{_label(chunk)}\n{body}" if body else ""
    except Exception as e:
        log.warning("summarize.map.error", source_id=chunk.source_id, error=str(e))
        return ""


def _reduce(notes: list[str], scope: str, llm_cfg: LLMConfig) -> str:
    if not notes:
        return "(no material to summarise)"
    joined = "\n\n".join(notes)
    prompt = REDUCE_PROMPT.format(scope=scope, notes=joined)
    try:
        resp = ollama.chat(
            model=llm_cfg.local_model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.2},
        )
        return str(resp.get("message", {}).get("content", "")).strip()
    except Exception as e:
        log.warning("summarize.reduce.error", error=str(e))
        return "(reduce step failed)"


def _fetch_filter(store: QdrantStore, where: dict[str, Any], limit: int) -> list[Chunk]:
    """Walk Qdrant by payload filter and hydrate chunks. Bounded by ``limit``."""
    from akb.retrieve.hybrid import _client_filter as _to_qdrant_filter
    from akb.store.qdrant_store import _hydrate

    qfilter = _to_qdrant_filter(where)
    chunks: list[Chunk] = []
    offset: Any = None
    client = store.client
    while True:
        batch, offset = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=qfilter,
            with_payload=True,
            with_vectors=False,
            limit=min(256, limit - len(chunks)),
            offset=offset,
        )
        for p in batch:
            chunks.append(_hydrate(p.payload or {}))
            if len(chunks) >= limit:
                return chunks
        if offset is None:
            break
    return chunks


def summarize_source(
    source_id: str,
    *,
    limit: int = 200,
    store: QdrantStore | None = None,
) -> SummaryResult:
    """Map-reduce every chunk of one source into a structured brief."""
    store = store or get_store()
    chunks = store.fetch_chunks_for_sources([source_id], limit_per_source=limit)
    return _run(chunks, scope=f"source `{source_id}`")


def summarize_filter(
    where: dict[str, Any],
    *,
    limit: int = 200,
    store: QdrantStore | None = None,
) -> SummaryResult:
    """Same as :func:`summarize_source` but the scope is anything matching ``where``."""
    store = store or get_store()
    chunks = _fetch_filter(store, where, limit=limit)
    return _run(chunks, scope=f"filter {where}")


def _run(chunks: list[Chunk], *, scope: str) -> SummaryResult:
    llm = load_settings().llm
    if not chunks:
        return SummaryResult(text="(no chunks matched)", scope=scope, model=llm.local_model)
    notes: list[str] = []
    for c in chunks:
        body = _map_one(c, llm)
        if body:
            notes.append(body)
    summary = _reduce(notes, scope=scope, llm_cfg=llm)
    log.info("summarize.done", scope=scope, n_chunks=len(chunks), n_notes=len(notes))
    return SummaryResult(
        text=summary, map_notes=notes, chunks=chunks, scope=scope, model=llm.local_model
    )


def summarize_dispatch(target: str) -> SummaryResult:
    """CLI-friendly: ``target`` may be ``source:<id>``, ``tag:<name>``, or
    ``type:<source_type>``."""
    if target.startswith("source:"):
        return summarize_source(target.split(":", 1)[1])
    if target.startswith("tag:"):
        return summarize_filter({"tags": [target.split(":", 1)[1]]})
    if target.startswith("type:"):
        return summarize_filter({"source_type": target.split(":", 1)[1]})
    # Fallback: treat as a source_id
    return summarize_source(target)


_ = Iterable  # re-exported for typing convenience
