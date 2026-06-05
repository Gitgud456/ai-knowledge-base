"""Secret scrubber — coverage of the high-confidence patterns + each policy."""

from __future__ import annotations

from akb.config import IngestConfig
from akb.ingest.scrubber import find_secrets, redact_text, scrub_chunks
from akb.schemas import Chunk, SourceType


def _c(text: str) -> Chunk:
    return Chunk(source_id="t", source_type=SourceType.txt, text=text)


def test_finds_aws_access_key() -> None:
    hits = find_secrets("creds: AKIAIOSFODNN7EXAMPLE end")
    assert any(h.name == "aws_access_key" for h in hits)


def test_finds_openai_key() -> None:
    hits = find_secrets("API: sk-proj-aaaaaaaaaaaaaaaaaaaaaaaa keep")
    assert any(h.name == "openai_key" for h in hits)


def test_finds_anthropic_key() -> None:
    hits = find_secrets("ANTHROPIC: sk-ant-api03-AbCdEfGhIjKlMnOpQrStUv next")
    assert any(h.name == "anthropic_key" for h in hits)


def test_finds_github_token() -> None:
    hits = find_secrets("token=ghp_abcdef0123456789abcdef0123456789abcdef done")
    assert any(h.name == "github_token" for h in hits)


def test_finds_google_api_key() -> None:
    hits = find_secrets("GOOGLE_API_KEY=AIzaSyCuyUDyjGxMBhAicD4WZBxwnkXX8aCBpUE done")
    assert any(h.name == "google_api_key" for h in hits)


def test_finds_private_key_block() -> None:
    text = (
        "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA...\n-----END RSA PRIVATE KEY-----"
    )
    hits = find_secrets(text)
    assert any(h.name == "private_key_block" for h in hits)


def test_redact_replaces_with_marker() -> None:
    text = "key=AKIAIOSFODNN7EXAMPLE rest"
    out = redact_text(text)
    assert "AKIAIOSFODNN7EXAMPLE" not in out
    assert "<REDACTED:aws_access_key>" in out


def test_policy_off_no_op() -> None:
    chunks = [_c("AKIAIOSFODNN7EXAMPLE")]
    out = scrub_chunks(chunks, IngestConfig(scrub_secrets="off"))
    assert out[0].text == "AKIAIOSFODNN7EXAMPLE"


def test_policy_block_drops_chunk() -> None:
    chunks = [_c("AKIAIOSFODNN7EXAMPLE"), _c("clean text")]
    out = scrub_chunks(chunks, IngestConfig(scrub_secrets="block"))
    assert len(out) == 1
    assert out[0].text == "clean text"


def test_policy_redact_replaces_in_place() -> None:
    chunks = [_c("key=AKIAIOSFODNN7EXAMPLE done")]
    out = scrub_chunks(chunks, IngestConfig(scrub_secrets="redact"))
    assert "AKIAIOSFODNN7EXAMPLE" not in out[0].text
    assert "<REDACTED:" in out[0].text


def test_policy_warn_passes_through() -> None:
    chunks = [_c("key=AKIAIOSFODNN7EXAMPLE done")]
    out = scrub_chunks(chunks, IngestConfig(scrub_secrets="warn"))
    assert "AKIAIOSFODNN7EXAMPLE" in out[0].text
