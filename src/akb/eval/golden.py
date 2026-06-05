"""Golden-set loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class GoldenItem:
    id: str
    question: str
    expected_sources: list[str] = field(default_factory=list)
    expected_answer_contains: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    notes: str = ""


def load_golden(path: Path) -> list[GoldenItem]:
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    items = data.get("items", [])
    out: list[GoldenItem] = []
    for raw in items:
        out.append(
            GoldenItem(
                id=str(raw["id"]),
                question=str(raw["question"]),
                expected_sources=[str(s) for s in raw.get("expected_sources", []) or []],
                expected_answer_contains=[
                    str(s) for s in raw.get("expected_answer_contains", []) or []
                ],
                tags=[str(t) for t in raw.get("tags", []) or []],
                notes=str(raw.get("notes", "")),
            )
        )
    return out
