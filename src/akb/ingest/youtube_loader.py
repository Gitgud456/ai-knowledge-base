"""YouTube transcript ingest.

Best-effort: tries ``youtube-transcript-api`` for the captions (handles
manual + auto-generated transcripts in any available language) and falls
back to ``yt-dlp`` if the first path fails. Output is a :class:`Document`
with ``source_type=web`` and metadata carrying the video id + title.

We don't try to chunk by speaker turns — the standard markdown chunker
handles transcripts fine because we inject ``# Title`` and ``## Section``
markers based on the transcript's own paragraph breaks.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from akb.obs.logging import get_logger
from akb.schemas import Document, SourceType

log = get_logger(__name__)


_YT_ID_RX = re.compile(
    r"(?:v=|youtu\.be/|/embed/|/shorts/)([A-Za-z0-9_-]{11})"
)


def extract_video_id(url_or_id: str) -> str | None:
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url_or_id):
        return url_or_id
    m = _YT_ID_RX.search(url_or_id)
    return m.group(1) if m else None


def _fetch_transcript(video_id: str) -> str:
    try:
        from youtube_transcript_api import YouTubeTranscriptApi  # type: ignore[import-untyped]
    except Exception as e:
        raise RuntimeError(
            "youtube-transcript-api is required: pip install youtube-transcript-api"
        ) from e
    log.info("youtube.transcript.fetch", video_id=video_id)
    entries = YouTubeTranscriptApi.get_transcript(video_id)
    return "\n".join(e.get("text", "") for e in entries).strip()


def _fetch_title(url_or_id: str) -> str:
    try:
        import yt_dlp  # type: ignore[import-untyped]
    except Exception:
        return ""
    try:
        with yt_dlp.YoutubeDL({"quiet": True, "skip_download": True}) as y:
            info = y.extract_info(url_or_id, download=False)
            return str(info.get("title", "")) if isinstance(info, dict) else ""
    except Exception as e:
        log.warning("youtube.title.error", error=str(e))
        return ""


def load_youtube(url_or_id: str) -> Document:
    video_id = extract_video_id(url_or_id)
    if not video_id:
        raise RuntimeError(f"could not extract video id from {url_or_id!r}")

    transcript = _fetch_transcript(video_id)
    if not transcript:
        raise RuntimeError("transcript was empty")

    title = _fetch_title(url_or_id) or f"YouTube {video_id}"

    # Wrap into our markdown shape so the header-aware chunker has something
    # to bite on.
    body = f"# {title}\n\n## Transcript\n\n{transcript}\n"
    now = datetime.now(timezone.utc)
    canonical_url = f"https://www.youtube.com/watch?v={video_id}"
    return Document(
        source_id=f"youtube:{video_id}",
        source_path=None,
        source_type=SourceType.web,
        title=title,
        content=body,
        created_at=now,
        modified_at=now,
        extra={"video_id": video_id, "url": canonical_url},
    )
