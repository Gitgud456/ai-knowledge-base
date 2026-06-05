"""Time-aware retrieval — temporal hints in the query become a date filter
on the ``modified_at`` payload field.

We use ``dateparser`` for natural-language phrases ("last March", "this week",
"yesterday", "before 2023"). The parsed range is translated into the
``{"modified_at": {"gte": "...", "lte": "..."}}`` shape that ``hybrid._client_filter``
already understands.

Failure modes are silent: if a date can't be parsed, the query goes through
without a filter. The user can confirm whether time filtering kicked in by
inspecting ``state['time_hint']`` (surfaced in the "Why this answer" panel).
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

try:
    import dateparser
except Exception:  # pragma: no cover
    dateparser = None  # type: ignore[assignment]


_HINT_RX = re.compile(
    r"""
    \b(
        today | yesterday |
        (this|last|past|next)\ (week|month|quarter|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday) |
        (this|last|past|next)\ \w+ |       # "last March", "last summer"
        (in|during|before|after|since)\ [\w\s,]{2,30} |
        \d{1,2}\ ?(months?|weeks?|days?|years?)\ ago |
        (?:earlier\ )?this\ year | this\ month |
        \d{4}                              # bare year
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _to_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def _date_range(phrase: str) -> tuple[datetime, datetime] | None:
    """Resolve a phrase into a (start, end) UTC range. Returns None on failure."""
    if dateparser is None:
        return None
    base = dateparser.parse(
        phrase,
        settings={"PREFER_DATES_FROM": "past", "RETURN_AS_TIMEZONE_AWARE": True},
    )
    if base is None:
        return None
    base = base if base.tzinfo else base.replace(tzinfo=timezone.utc)

    low = phrase.lower()
    if "week" in low:
        start = base - timedelta(days=base.weekday())
        end = start + timedelta(days=7)
    elif "month" in low:
        start = base.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if start.month == 12:
            end = start.replace(year=start.year + 1, month=1)
        else:
            end = start.replace(month=start.month + 1)
    elif "year" in low or re.fullmatch(r"\d{4}", low.strip()):
        start = base.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        end = start.replace(year=start.year + 1)
    elif "today" in low or "yesterday" in low or "ago" in low:
        start = base.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
    else:
        start = base.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
    return start, end


def _open_range(phrase: str) -> tuple[datetime | None, datetime | None]:
    """Handle 'before X' / 'after X' / 'since X' patterns."""
    if dateparser is None:
        return None, None
    low = phrase.lower().strip()
    for kw, half in (("before ", "lt"), ("after ", "gt"), ("since ", "gt")):
        if low.startswith(kw):
            target = dateparser.parse(
                low[len(kw):].strip(),
                settings={"PREFER_DATES_FROM": "past", "RETURN_AS_TIMEZONE_AWARE": True},
            )
            if target is None:
                return None, None
            if target.tzinfo is None:
                target = target.replace(tzinfo=timezone.utc)
            return (None, target) if half == "lt" else (target, None)
    return None, None


def extract_hint(text: str) -> str | None:
    """Return the first temporal phrase found in ``text`` (or None)."""
    m = _HINT_RX.search(text)
    return m.group(0).strip() if m else None


def build_time_filter(query: str) -> tuple[dict | None, str]:
    """Parse ``query``, return a Qdrant-style filter dict and a human label.

    The filter has shape::

        {"modified_at": {"gte": "<iso>", "lte": "<iso>"}}

    Any of gte/lte may be omitted (open-ended). The string label is suitable
    for the "Why this answer" panel.
    """
    hint = extract_hint(query)
    if not hint:
        return None, ""

    gte, lte = _open_range(hint)
    if gte is None and lte is None:
        rng = _date_range(hint)
        if rng is None:
            return None, ""
        gte, lte = rng

    bounds: dict[str, str] = {}
    parts: list[str] = []
    if gte is not None:
        bounds["gte"] = _to_iso(gte)
        parts.append(f"since {gte.date().isoformat()}")
    if lte is not None:
        bounds["lte"] = _to_iso(lte)
        parts.append(f"before {lte.date().isoformat()}")
    if not bounds:
        return None, ""

    label = f"{hint!r} → " + ", ".join(parts)
    return {"modified_at": bounds}, label
