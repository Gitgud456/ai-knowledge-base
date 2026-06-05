"""Typed configuration loader.

Precedence (low → high):
  1. configs/default.yaml
  2. configs/local.yaml (gitignored, optional)
  3. Environment variables (AKB_* with `__` for nesting; .env auto-loaded)

Usage:
    from akb.config import load_settings
    settings = load_settings()
    settings.retrieve.top_k
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"
LOCAL_CONFIG = REPO_ROOT / "configs" / "local.yaml"


def _expand_env(value: Any) -> Any:
    """Recursively expand ${VAR} placeholders against os.environ."""
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in overlay.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return _expand_env(data)


class PathsConfig(BaseModel):
    vault: Path
    data_dir: Path = Path("./data")
    qdrant_dir: Path = Path("./data/qdrant")
    bm25_dir: Path = Path("./data/bm25")
    session_db: Path = Path("./data/session_history.db")
    ingest_state_db: Path = Path("./data/ingest_state.db")
    prompts_dir: Path = Path("./src/akb/prompts")

    @field_validator("*", mode="before")
    @classmethod
    def _to_path(cls, v: Any) -> Path:
        return Path(v) if not isinstance(v, Path) else v

    @field_validator(
        "data_dir", "qdrant_dir", "bm25_dir", "session_db", "ingest_state_db", "prompts_dir",
        mode="after",
    )
    @classmethod
    def _resolve_against_repo(cls, v: Path) -> Path:
        """Anchor relative paths to the repo root, not the CWD.

        Without this, running ``akb`` from a different working directory silently
        creates a new index in that cwd instead of reusing the existing one.
        Absolute paths pass through unchanged.
        """
        if v.is_absolute():
            return v
        return (REPO_ROOT / v).resolve()


class LLMConfig(BaseModel):
    local_model: str = "llama3:8b-instruct-q4_K_M"
    context_model: str = "llama3:8b-instruct-q4_K_M"
    deep_provider: str = "gemini"
    gemini_model: str = "gemini-1.5-flash"
    ollama_host: str = "http://localhost:11434"
    temperature: float = 0.2
    max_tokens: int = 2048


class EmbedConfig(BaseModel):
    model: str = "BAAI/bge-m3"
    dim: int = 1024
    normalize: bool = True
    batch_size: int = 32
    use_sparse: bool = True
    # Binary quantization in Qdrant: ~32x smaller, ~40x faster, with
    # oversampling=2 + rescore the recall loss is typically <1 point.
    # Toggling this requires rebuilding the collection (drop + reindex).
    binary_quantization: bool = False
    binary_oversampling: float = 2.0


class IngestConfig(BaseModel):
    chunk_size: int = 1200
    chunk_overlap: int = 200
    headers_to_split: list[list[str]] = Field(
        default_factory=lambda: [["#", "h1"], ["##", "h2"], ["###", "h3"], ["####", "h4"]]
    )
    batch_size: int = 64
    semantic_chunker: bool = False
    contextual_retrieval: bool = True
    # How many chunks per Ollama call in contextualizer batch mode. ~8-16 is the
    # sweet spot for an 8B model — bigger batches drop output quality, smaller
    # ones lose the speedup.
    context_batch_size: int = 8
    late_chunking: bool = False
    skip_dirs: list[str] = Field(default_factory=lambda: [".obsidian", ".trash"])
    attachment_dirs: list[str] = Field(default_factory=lambda: ["attachments"])
    # Secret scrubbing: drop chunks (or warn) that look like leaked credentials.
    # 'block' deletes the chunk entirely, 'redact' replaces the secret in-place,
    # 'warn' logs and lets it through, 'off' disables scanning.
    scrub_secrets: str = "redact"


class RetrieveConfig(BaseModel):
    n_results: int = 50
    top_k: int = 8
    rrf_k: int = 60
    dense_weight: float = 1.0
    sparse_weight: float = 1.0
    use_reranker: bool = True
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_top_n: int = 50
    use_colbert_rerank: bool = False
    graph_expand: bool = True
    graph_hops: int = 1
    graph_expand_limit: int = 5
    # Recency weighting: multiplies rerank score by exp(-age_days/half_life).
    # 0.0 disables. half_life of 180d => a 6-month-old note keeps ~50% of its
    # rerank score relative to today's. Higher weights bias harder to recent.
    recency_weight: float = 0.0
    recency_half_life_days: float = 180.0


class AgentConfig(BaseModel):
    framework: str = "langgraph"
    enable_router: bool = True
    enable_critic: bool = True
    enable_web_fallback: bool = True
    web_tool: str = "duckduckgo"
    max_critic_iterations: int = 1
    history_window: int = 8
    # Cross-session long-term memory.
    enable_memory: bool = True
    memory_top_k: int = 3
    memory_remember_threshold: int = 16   # skip short answers (small talk)
    # Time-aware retrieval: parse temporal hints ("last March", "this week")
    # and add a Qdrant filter on modified_at.
    enable_time_aware: bool = True


class MentorConfig(BaseModel):
    initial_recall: int = 20
    initial_top_k: int = 10
    topic_recall: int = 10
    topic_top_k: int = 5
    history_window: int = 6


class EvalConfig(BaseModel):
    golden_path: Path = Path("./tests/golden/golden_set.yaml")
    judge_model: str = "llama3:8b-instruct-q4_K_M"
    fail_on_regression_pct: float = 5.0


class ObsConfig(BaseModel):
    log_level: str = "INFO"
    json_logs: bool = True
    enable_tracing: bool = False
    langfuse_host: str = "http://localhost:3000"


class Settings(BaseSettings):
    """Top-level typed config. Override any nested field via env (`AKB_RETRIEVE__TOP_K=12`)."""

    paths: PathsConfig
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embed: EmbedConfig = Field(default_factory=EmbedConfig)
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    retrieve: RetrieveConfig = Field(default_factory=RetrieveConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    mentor: MentorConfig = Field(default_factory=MentorConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    obs: ObsConfig = Field(default_factory=ObsConfig)

    model_config = SettingsConfigDict(
        env_prefix="AKB_",
        env_nested_delimiter="__",
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def load_settings() -> Settings:
    """Load + validate the merged config. Cached for the lifetime of the process."""
    merged = _deep_merge(_load_yaml(DEFAULT_CONFIG), _load_yaml(LOCAL_CONFIG))
    return Settings(**merged)


def reset_settings_cache() -> None:
    """Drop the cached settings — useful in tests."""
    load_settings.cache_clear()
