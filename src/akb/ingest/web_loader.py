"""Web URL ingest via ``trafilatura``.

Fetches an article, extracts main content (drops nav/comments/ads), and packages
it as a :class:`Document` with ``source_type=web``. The URL becomes the
``source_id`` so re-ingesting an unchanged page is idempotent (deterministic
chunk ids + the index stamp see no diff).

For PDFs / EPUBs hosted at a URL, prefer downloading them and ingesting via
the regular ``akb ingest <path>`` pipeline — trafilatura is HTML-only.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone

from akb.obs.logging import get_logger
from akb.schemas import Document, SourceType

log = get_logger(__name__)

_TITLE_RX = re.compile(r"<title>([^<]+)</title>", re.IGNORECASE)


def _slug(url: str) -> str:
    h = hashlib.sha1(url.encode()).hexdigest()[:10]
    return h


def load_url(url: str) -> Document:
    try:
        import trafilatura  # type: ignore[import-untyped]
    except Exception as e:
        raise RuntimeError(
            "trafilatura is required for web ingest: pip install trafilatura"
        ) from e

    log.info("web.fetch", url=url)
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        raise RuntimeError(f"could not fetch {url}")

    # Try the markdown output first (preserves headings → good for our chunker);
    # fall back to plain text.
    text = trafilatura.extract(
        downloaded,
        output_format="markdown",
        include_links=False,
        include_images=False,
        include_tables=True,
        with_metadata=False,
    )
    if not text:
        text = trafilatura.extract(downloaded) or ""
    if not text.strip():
        raise RuntimeError(f"no extractable content at {url}")

    title = ""
    meta = trafilatura.extract_metadata(downloaded)
    if meta is not None:
        title = getattr(meta, "title", "") or ""
    if not title:
        m = _TITLE_RX.search(downloaded or "")
        if m:
            title = m.group(1).strip()
    if not title:
        title = url

    now = datetime.now(timezone.utc)
    return Document(
        source_id=f"web:{url}",
        source_path=None,
        source_type=SourceType.web,
        title=title,
        content=text,
        created_at=now,
        modified_at=now,
        extra={"url": url, "slug": _slug(url)},
    )
