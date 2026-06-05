"""Late chunking (Günther et al., 2024 — arXiv:2409.04701).

The idea: embed the *whole* document with a long-context embedder once, then
mean-pool the token embeddings *per chunk span* afterwards. Every chunk vector
carries global document context — no LLM call required, unlike contextual
retrieval (Phase 4). Cheaper at index time, slightly less semantic lift.

Implementation here uses BGE-M3 in token-output mode (returns hidden states).
We only run this when the document is long enough that orphan-chunk problems
actually matter; short notes go through the normal embedder.

Opt-in via ``ingest.late_chunking: true`` in config. Off by default because
the inference cost per long doc is non-trivial.
"""

from __future__ import annotations

from dataclasses import dataclass

from akb.config import EmbedConfig, load_settings
from akb.embed.providers import EmbeddedBatch, SparseVec
from akb.schemas import Chunk, Document

# Activation threshold — docs shorter than this go through normal embedding.
DEFAULT_MIN_CHARS = 4000


@dataclass
class LateChunkResult:
    chunks: list[Chunk]
    embeddings: EmbeddedBatch


def _segment_offsets(text: str, chunks: list[Chunk]) -> list[tuple[int, int]]:
    """Best-effort char-span recovery. We find each chunk's first occurrence
    in the original document text; if not found (e.g. chunk was contextualized),
    we fall back to a uniform partition.
    """
    spans: list[tuple[int, int]] = []
    cursor = 0
    for c in chunks:
        i = text.find(c.text[:64], cursor) if c.text else -1
        if i < 0:
            # uniform fallback
            n = len(chunks) or 1
            chunk_len = max(1, len(text) // n)
            for k in range(n):
                spans.append((k * chunk_len, min(len(text), (k + 1) * chunk_len)))
            return spans
        end = i + len(c.text)
        spans.append((i, end))
        cursor = end
    return spans


def late_embed(
    document: Document,
    chunks: list[Chunk],
    cfg: EmbedConfig | None = None,
    min_chars: int = DEFAULT_MIN_CHARS,
) -> EmbeddedBatch | None:
    """Return per-chunk embeddings from a single long-context document pass.

    Returns ``None`` if late chunking is not applicable (doc too short, or the
    underlying model doesn't expose token-level outputs). Caller falls back to
    normal :func:`akb.embed.providers.get_embedder`.
    """
    cfg = cfg or load_settings().embed
    if len(document.content) < min_chars:
        return None
    try:
        from FlagEmbedding import BGEM3FlagModel  # type: ignore[import-untyped]
        import numpy as np
    except Exception:
        return None

    model = BGEM3FlagModel(cfg.model, use_fp16=True)
    out = model.encode(  # type: ignore[attr-defined]
        [document.content],
        batch_size=1,
        return_dense=False,
        return_sparse=False,
        return_colbert_vecs=True,
    )
    token_embs = out.get("colbert_vecs", [None])[0]
    if token_embs is None:
        return None

    # rough char→token mapping by uniform proportion (cheap, good enough for
    # well-tokenized prose). For higher fidelity, swap in the tokenizer offsets.
    n_tokens = len(token_embs)
    n_chars = max(1, len(document.content))
    spans = _segment_offsets(document.content, chunks)

    dense: list[list[float]] = []
    for char_start, char_end in spans:
        tok_start = int(char_start / n_chars * n_tokens)
        tok_end = max(tok_start + 1, int(char_end / n_chars * n_tokens))
        slice_ = token_embs[tok_start:tok_end]
        vec = np.mean(slice_, axis=0) if len(slice_) else np.zeros(token_embs.shape[1])
        if cfg.normalize:
            norm = float(np.linalg.norm(vec)) + 1e-12
            vec = vec / norm
        dense.append(vec.tolist())

    return EmbeddedBatch(
        dense=dense,
        sparse=[SparseVec(indices=[], values=[]) for _ in chunks],
    )
