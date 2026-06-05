"""Secret scrubbing at ingest.

Personal vaults accumulate things you don't want in an LLM context window:
pasted API keys, AWS credentials, SSH private keys, OAuth tokens from
copy-pasted curl calls. Once those land in Qdrant they leak into every prompt
that touches a neighbouring chunk.

This module scans each chunk for high-confidence secret patterns *before*
upsert and applies one of four policies:

  * ``off``     — disabled (skip the scan entirely)
  * ``warn``    — log and pass the chunk through unchanged
  * ``redact``  — replace each detected secret with ``<REDACTED:<kind>>`` (default)
  * ``block``   — drop the entire chunk

Patterns are intentionally conservative — false negatives are fine here (the
user can audit), but a false positive that mangles a code sample is a worse UX
than a leaked key in a personal vault. Add patterns as you find blind spots.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from akb.config import IngestConfig, load_settings
from akb.obs.logging import get_logger
from akb.schemas import Chunk

log = get_logger(__name__)


@dataclass(frozen=True)
class Pattern:
    name: str
    regex: re.Pattern[str]


# High-confidence patterns (high specificity, low false-positive rate)
PATTERNS: list[Pattern] = [
    Pattern(
        "aws_access_key",
        re.compile(r"\b(AKIA|ASIA)[0-9A-Z]{16}\b"),
    ),
    Pattern(
        "aws_secret_key",
        # Heuristic: 40-char base64-ish next to "aws_secret" mention; OR
        # standalone but with stronger context. Skip pure standalone to avoid FPs.
        re.compile(r"(?i)aws[_\- ]?secret[^=:\n]{0,40}[=:]\s*['\"]?([A-Za-z0-9/+=]{40})['\"]?"),
    ),
    Pattern(
        "openai_key",
        re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b"),
    ),
    Pattern(
        "anthropic_key",
        re.compile(r"\bsk-ant-(?:api|admin)\d{2}-[A-Za-z0-9_-]{20,}\b"),
    ),
    Pattern(
        "github_token",
        re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,}\b"),
    ),
    Pattern(
        "google_api_key",
        re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
    ),
    Pattern(
        "slack_token",
        re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b"),
    ),
    Pattern(
        "private_key_block",
        re.compile(
            r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY( BLOCK)?-----[\s\S]*?-----END "
            r"(?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY( BLOCK)?-----",
        ),
    ),
    Pattern(
        "jwt",
        # eyJ-prefixed JWT: two base64url segments + signature. Reasonable
        # FP rate is acceptable since people rarely paste lookalike strings.
        re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
    ),
]


@dataclass
class ScrubResult:
    name: str
    span: tuple[int, int]


def find_secrets(text: str) -> list[ScrubResult]:
    """Return the list of detected secrets in ``text``, ordered by position."""
    hits: list[ScrubResult] = []
    for pat in PATTERNS:
        for m in pat.regex.finditer(text):
            hits.append(ScrubResult(name=pat.name, span=m.span()))
    hits.sort(key=lambda r: r.span[0])
    return hits


def redact_text(text: str, hits: list[ScrubResult] | None = None) -> str:
    hits = hits if hits is not None else find_secrets(text)
    if not hits:
        return text
    out: list[str] = []
    cursor = 0
    for h in hits:
        s, e = h.span
        if s < cursor:  # overlapping match — keep the first
            continue
        out.append(text[cursor:s])
        out.append(f"<REDACTED:{h.name}>")
        cursor = e
    out.append(text[cursor:])
    return "".join(out)


def scrub_chunks(
    chunks: Iterable[Chunk],
    cfg: IngestConfig | None = None,
) -> list[Chunk]:
    """Apply the configured scrubbing policy. See module docstring for modes."""
    cfg = cfg or load_settings().ingest
    policy = (cfg.scrub_secrets or "off").lower()
    if policy == "off":
        return list(chunks)

    out: list[Chunk] = []
    blocked = 0
    redacted = 0
    for chunk in chunks:
        hits = find_secrets(chunk.text)
        if not hits:
            out.append(chunk)
            continue
        kinds = sorted({h.name for h in hits})
        log.warning(
            "scrub.hit",
            source_id=chunk.source_id,
            chunk_index=chunk.chunk_index,
            kinds=kinds,
            policy=policy,
        )
        if policy == "warn":
            out.append(chunk)
        elif policy == "block":
            blocked += 1
            continue
        else:  # redact (default)
            chunk.text = redact_text(chunk.text, hits)
            # The deterministic chunk_id was derived from the now-redacted text
            # would still match what the validator computed at construction —
            # we mutate text in-place AFTER construction so the id is stable
            # against the *original* text. That's intentional: re-running with
            # `scrub_secrets=off` shouldn't shift IDs.
            if chunk.contextualized_text:
                chunk.contextualized_text = redact_text(chunk.contextualized_text)
            redacted += 1
            out.append(chunk)
    if blocked or redacted:
        log.info("scrub.summary", blocked=blocked, redacted=redacted)
    return out
