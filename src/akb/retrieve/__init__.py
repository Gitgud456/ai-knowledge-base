"""Retrieval pipeline: query transforms → hybrid → graph expansion → rerank.

Phase 2 lands hybrid + RRF. Phase 3 lands the BGE reranker. Phase 1 already exposes
graph edges in chunk payloads for use by graph_expand.
"""
