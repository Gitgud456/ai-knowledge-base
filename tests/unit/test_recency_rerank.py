"""Recency-weighted rerank: timestamp-aware multiplier on cross-encoder scores."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from akb.config import RetrieveConfig
from akb.retrieve.rerank import _age_days, _apply_recency
from akb.schemas import Chunk, RetrievedChunk, SourceType


def _rc(score: float, days_old: int) -> RetrievedChunk:
    ts = (datetime.now(timezone.utc) - timedelta(days=days_old)).isoformat()
    chunk = Chunk(
        source_id="t",
        source_type=SourceType.obsidian,
        text="x",
        metadata={"modified_at": ts},
    )
    return RetrievedChunk(chunk=chunk, rerank_score=score)


def test_age_days_basic() -> None:
    rc = _rc(1.0, days_old=10)
    age = _age_days(rc, datetime.now(timezone.utc))
    assert age is not None
    assert 9.5 <= age <= 10.5


def test_age_days_missing_timestamp() -> None:
    chunk = Chunk(source_id="t", source_type=SourceType.txt, text="x")
    rc = RetrievedChunk(chunk=chunk, rerank_score=1.0)
    assert _age_days(rc, datetime.now(timezone.utc)) is None


def test_apply_recency_no_effect_when_weight_zero() -> None:
    rc = _rc(1.0, days_old=365)
    _apply_recency([rc], weight=0.0, half_life_days=30.0)
    assert rc.rerank_score == 1.0


def test_apply_recency_decays_old_chunks() -> None:
    fresh = _rc(1.0, days_old=0)
    old = _rc(1.0, days_old=365)
    _apply_recency([fresh, old], weight=1.0, half_life_days=180.0)
    assert fresh.rerank_score is not None and fresh.rerank_score > 0.95
    assert old.rerank_score is not None and old.rerank_score < 0.5


def test_apply_recency_partial_weight_interpolates() -> None:
    full = _rc(1.0, days_old=180)
    partial = _rc(1.0, days_old=180)
    _apply_recency([full], weight=1.0, half_life_days=180.0)
    _apply_recency([partial], weight=0.5, half_life_days=180.0)
    # 180d at half_life=180 -> decay=0.5; weight=0.5 -> (1-0.5)*1 + 0.5*0.5 = 0.75
    assert full.rerank_score is not None and partial.rerank_score is not None
    assert full.rerank_score < partial.rerank_score


def test_apply_recency_passes_through_chunks_without_score() -> None:
    rc = _rc(1.0, days_old=10)
    rc.rerank_score = None
    _apply_recency([rc], weight=1.0, half_life_days=30.0)
    assert rc.rerank_score is None
