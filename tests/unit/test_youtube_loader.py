"""YouTube video-id extraction. Network paths are mocked separately."""

from __future__ import annotations

from akb.ingest.youtube_loader import extract_video_id


def test_bare_id() -> None:
    assert extract_video_id("dQw4w9WgXcQ") == "dQw4w9WgXcQ"


def test_watch_url() -> None:
    assert extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"


def test_short_url() -> None:
    assert extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"


def test_embed_url() -> None:
    assert extract_video_id("https://www.youtube.com/embed/dQw4w9WgXcQ") == "dQw4w9WgXcQ"


def test_shorts_url() -> None:
    assert extract_video_id("https://www.youtube.com/shorts/dQw4w9WgXcQ") == "dQw4w9WgXcQ"


def test_unrelated_string() -> None:
    assert extract_video_id("not a youtube url at all") is None


def test_query_extra_params() -> None:
    assert (
        extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PL123")
        == "dQw4w9WgXcQ"
    )
