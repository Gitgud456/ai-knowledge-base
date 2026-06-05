"""Pydantic v2 data models shared across the package.

Every piece of data that crosses module boundaries flows through one of these.
Keep them lean — store policy lives in services, schemas just describe shape.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class SourceType(str, Enum):
    obsidian = "obsidian"
    pdf = "pdf"
    epub = "epub"
    txt = "txt"
    manual = "manual"
    web = "web"


class Document(BaseModel):
    """A parsed source artifact prior to chunking."""

    model_config = ConfigDict(frozen=True)

    source_id: str = Field(description="Stable canonical ID for the source (path-relative for files).")
    source_path: Path | None = None
    source_type: SourceType
    title: str | None = None
    content: str
    frontmatter: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    wikilinks: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    created_at: datetime | None = None
    modified_at: datetime | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()


class Chunk(BaseModel):
    """A single retrieval unit. One Document → many Chunks."""

    model_config = ConfigDict(frozen=False)

    chunk_id: str = Field(default_factory=lambda: uuid4().hex)
    source_id: str
    source_type: SourceType
    text: str
    contextualized_text: str | None = Field(
        default=None,
        description="Anthropic-style situating prefix + text. Set by Phase 4 contextualizer.",
    )
    header_path: list[str] = Field(default_factory=list, description="Breadcrumb of H1>H2>H3 titles.")
    chunk_index: int = 0
    char_start: int | None = None
    char_end: int | None = None
    tags: list[str] = Field(default_factory=list)
    wikilinks: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def embed_text(self) -> str:
        """Text actually handed to the embedder (may include contextual prefix)."""
        return self.contextualized_text or self.text


class RetrievedChunk(BaseModel):
    """A chunk returned by retrieval, with its score(s)."""

    chunk: Chunk
    dense_score: float | None = None
    sparse_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None
    expanded_from: str | None = Field(
        default=None, description="If brought in by graph expansion, source chunk_id."
    )

    @property
    def final_score(self) -> float:
        return self.rerank_score or self.rrf_score or self.dense_score or 0.0


class Citation(BaseModel):
    source_id: str
    chunk_id: str
    snippet: str
    score: float


class Query(BaseModel):
    text: str
    metadata_filter: dict[str, Any] | None = None
    n_results: int | None = None
    top_k: int | None = None


class Answer(BaseModel):
    text: str
    citations: list[Citation] = Field(default_factory=list)
    used_tools: list[str] = Field(default_factory=list)
    model: str
    deep_mode: bool = False
    trace_id: str | None = None
    iterations: int = 1
