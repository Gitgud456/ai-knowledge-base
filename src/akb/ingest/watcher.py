"""Live vault watcher using `watchfiles` (Rust-backed, async, debounced).

Runs a background loop:
  * Listen for ``Change`` events on ``*.md`` files under the vault.
  * Debounce ~2s into a coalesced batch.
  * Run a targeted ``plan_sync`` over only the touched paths.

If you have a large vault, this is the difference between "I edited one note"
costing 1 embedding pass vs. 50 000.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

from watchfiles import Change, awatch

from akb.config import load_settings
from akb.ingest.sync import apply_sync, plan_sync


def _is_vault_md(path: Path, vault: Path, skip: set[str]) -> bool:
    if path.suffix.lower() != ".md":
        return False
    if not path.is_relative_to(vault):
        return False
    return not any(part.lower() in skip for part in path.parts)


async def watch_vault(
    *,
    vault: Path | None = None,
    debounce_ms: int = 2000,
) -> AsyncIterator[dict[str, int]]:
    """Async generator yielding sync results after each debounced batch."""
    settings = load_settings()
    vault = vault or settings.paths.vault
    skip = {d.lower() for d in settings.ingest.skip_dirs}

    async for changes in awatch(str(vault), debounce=debounce_ms):
        affected: set[Path] = set()
        for change, path_str in changes:
            p = Path(path_str)
            if _is_vault_md(p, vault, skip):
                affected.add(p)
            if change is Change.deleted:
                affected.add(p)  # plan_sync's set-diff handles the rest
        if not affected:
            continue
        # Targeted plan: we still ask plan_sync to compute deletes by full diff
        # (cheap — it only scans paths, not contents) but limit added/changed
        # to the watched batch.
        full = plan_sync(vault=vault)
        full.added = [p for p in full.added if p in affected]
        full.changed = [p for p in full.changed if p in affected]
        # Deletes always come from the full diff; partial info is unsafe.
        result = apply_sync(full, vault=vault)
        yield result


def run_watcher_forever(vault: Path | None = None) -> None:
    """Sync wrapper for CLI / Streamlit consumers that don't want async."""
    async def _main() -> None:
        async for _ in watch_vault(vault=vault):
            pass

    asyncio.run(_main())
