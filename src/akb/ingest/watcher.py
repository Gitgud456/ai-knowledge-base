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
            if _is_vault_md(p, vault, skip) or change is Change.deleted:
                affected.add(p)
        if not affected:
            continue
        # Targeted plan: inspect only the touched paths. plan_sync handles a
        # passed path that no longer exists by promoting it to a delete.
        # Note: this means a delete-on-watcher-restart of a file we already
        # know about won't be detected until the next full `akb sync` — but
        # that's the right perf trade-off for huge vaults.
        plan = plan_sync(vault=vault, restrict_paths=list(affected))
        if plan.total() == 0:
            continue
        result = apply_sync(plan, vault=vault)
        yield result


def run_watcher_forever(vault: Path | None = None) -> None:
    """Sync wrapper for CLI / Streamlit consumers that don't want async."""
    async def _main() -> None:
        async for _ in watch_vault(vault=vault):
            pass

    asyncio.run(_main())
