"""Ingest pipeline: loaders → chunkers → dedupe → contextualize → upsert.

Phase 1 wires loaders + chunkers. Phase 4 adds contextualizer. Phase 6 adds sync + watcher.
"""
