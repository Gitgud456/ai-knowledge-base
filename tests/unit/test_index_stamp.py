"""Index stamp + compatibility check using an in-memory Qdrant."""

from __future__ import annotations

from qdrant_client import QdrantClient

from akb.config import EmbedConfig
from akb.store.migrations import (
    IndexStamp,
    check_index_compatible,
    read_index_stamp,
    write_index_stamp,
)


def _client() -> QdrantClient:
    return QdrantClient(":memory:")


def test_first_run_writes_stamp_and_passes() -> None:
    cfg = EmbedConfig(model="m", dim=8, use_sparse=False, binary_quantization=False)
    client = _client()
    res = check_index_compatible(client, cfg)
    assert res.compatible
    live = read_index_stamp(client)
    assert live is not None
    assert live.embed_model == "m"
    assert live.embed_dim == 8


def test_model_mismatch_fails() -> None:
    client = _client()
    write_index_stamp(client, EmbedConfig(model="a", dim=8, use_sparse=False))
    new_cfg = EmbedConfig(model="b", dim=8, use_sparse=False)
    res = check_index_compatible(client, new_cfg)
    assert not res.compatible
    assert "embed model mismatch" in res.reason


def test_dim_mismatch_fails() -> None:
    client = _client()
    write_index_stamp(client, EmbedConfig(model="a", dim=4, use_sparse=False))
    new_cfg = EmbedConfig(model="a", dim=8, use_sparse=False)
    res = check_index_compatible(client, new_cfg)
    assert not res.compatible
    assert "dim" in res.reason


def test_quantization_mismatch_fails() -> None:
    client = _client()
    write_index_stamp(client, EmbedConfig(model="a", dim=8, binary_quantization=False))
    new_cfg = EmbedConfig(model="a", dim=8, binary_quantization=True)
    res = check_index_compatible(client, new_cfg)
    assert not res.compatible
    assert "binary_quantization" in res.reason


def test_stamp_from_payload_handles_missing_keys() -> None:
    s = IndexStamp.from_payload({})
    assert s.embed_model == ""
    assert s.embed_dim == 0
