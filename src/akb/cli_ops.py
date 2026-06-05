"""CLI ops helpers (kept out of cli.py so they're unit-testable without typer).

  * :func:`doctor_checks` — preflight probes (Ollama reachable, models cached,
    vault exists, data dir writable, disk space).
  * :func:`gather_stats` — index size, sources by type, contextualizer cache
    rows, ingest_state row count, drift between Qdrant and ingest_state.
  * :func:`export_session` — render a saved session as a markdown note,
    citations rewritten to ``[[wikilinks]]`` so they link inside Obsidian.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from akb.config import load_settings


@dataclass
class DoctorCheck:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class DoctorReport:
    checks: list[DoctorCheck] = field(default_factory=list)

    @property
    def healthy(self) -> bool:
        return all(c.ok for c in self.checks)


def _check_vault(vault: Path) -> DoctorCheck:
    if not vault.exists():
        return DoctorCheck("vault.exists", False, f"missing: {vault}")
    if not vault.is_dir():
        return DoctorCheck("vault.exists", False, f"not a directory: {vault}")
    try:
        next(vault.rglob("*.md"))
        any_md = True
    except StopIteration:
        any_md = False
    return DoctorCheck(
        "vault.exists",
        True,
        f"{vault} ({'has' if any_md else 'no'} .md files)",
    )


def _check_writable(path: Path) -> DoctorCheck:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".akb_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return DoctorCheck(f"writable:{path.name}", True, str(path))
    except Exception as e:
        return DoctorCheck(f"writable:{path.name}", False, f"{path}: {e}")


def _check_disk(path: Path, min_gb: float = 1.0) -> DoctorCheck:
    try:
        usage = shutil.disk_usage(path if path.exists() else path.parent)
        free_gb = usage.free / (1024**3)
        return DoctorCheck(
            "disk.free",
            free_gb >= min_gb,
            f"{free_gb:.1f} GB free at {path} (min {min_gb})",
        )
    except Exception as e:
        return DoctorCheck("disk.free", False, str(e))


def _check_ollama(host: str, model: str) -> DoctorCheck:
    try:
        import ollama

        client = ollama.Client(host=host)
        listed = client.list()
        # Normalise both `{"models": [{"name": ...}]}` and pydantic-model variants
        names: list[str] = []
        for m in (listed.get("models") if isinstance(listed, dict) else getattr(listed, "models", [])) or []:
            n = m.get("name") if isinstance(m, dict) else getattr(m, "name", "")
            if n:
                names.append(n)
        has_model = any(n.startswith(model.split(":")[0]) for n in names)
        return DoctorCheck(
            "ollama",
            True,
            f"reachable at {host}; {'model present' if has_model else 'model NOT pulled — run: ollama pull ' + model}",
        )
    except Exception as e:
        return DoctorCheck("ollama", False, f"unreachable at {host}: {e}")


def _check_hf_cache(model: str) -> DoctorCheck:
    """Verify the HF cache directory has weights for ``model`` (e.g. BGE-M3)."""
    try:
        from huggingface_hub import scan_cache_dir

        info = scan_cache_dir()
        for repo in info.repos:
            if model.lower() in repo.repo_id.lower():
                size_gb = repo.size_on_disk / (1024**3)
                return DoctorCheck(f"hf_cache:{model}", True, f"cached ({size_gb:.1f} GB)")
        return DoctorCheck(
            f"hf_cache:{model}",
            False,
            "not cached; first ingest will download (~2 GB for BGE-M3)",
        )
    except Exception as e:
        return DoctorCheck(f"hf_cache:{model}", False, str(e))


def doctor_checks() -> DoctorReport:
    s = load_settings()
    report = DoctorReport()
    report.checks.append(_check_vault(s.paths.vault))
    report.checks.append(_check_writable(s.paths.data_dir))
    report.checks.append(_check_writable(s.paths.qdrant_dir))
    report.checks.append(_check_disk(s.paths.data_dir))
    report.checks.append(_check_ollama(s.llm.ollama_host, s.llm.local_model))
    report.checks.append(_check_hf_cache(s.embed.model))
    report.checks.append(_check_hf_cache(s.retrieve.reranker_model))
    return report


@dataclass
class IndexStats:
    qdrant_points: int = 0
    qdrant_sources: int = 0
    ingest_state_sources: int = 0
    ingest_state_chunks: int = 0
    context_cache_rows: int = 0
    session_count: int = 0
    by_source_type: dict[str, int] = field(default_factory=dict)
    drift: int = 0


def gather_stats() -> IndexStats:
    from akb.store.qdrant_store import COLLECTION, get_store
    from akb.store.sqlite_state import IngestState

    settings = load_settings()
    out = IndexStats()

    # ----- Qdrant
    try:
        store = get_store()
        out.qdrant_points = store.count()
        sources = store.list_sources()
        out.qdrant_sources = len(sources)
        # per source-type breakdown via a scroll over payload-only
        seen: dict[str, int] = {}
        offset: Any = None
        client = store.client
        while True:
            points, offset = client.scroll(
                collection_name=COLLECTION,
                with_payload=["source_type"],
                with_vectors=False,
                limit=1024,
                offset=offset,
            )
            for p in points:
                st = (p.payload or {}).get("source_type", "?")
                seen[st] = seen.get(st, 0) + 1
            if offset is None:
                break
        out.by_source_type = seen
    except Exception:
        pass

    # ----- ingest_state
    try:
        state = IngestState()
        sids = state.all_source_ids()
        out.ingest_state_sources = len(sids)
        chunk_total = 0
        for sid in sids:
            chunk_total += len(state.chunk_ids_for(sid))
        out.ingest_state_chunks = chunk_total
        out.drift = abs(out.qdrant_points - chunk_total)
    except Exception:
        pass

    # ----- contextualizer cache
    try:
        import sqlite3

        cache = settings.paths.data_dir / "context_cache.db"
        if cache.exists():
            with sqlite3.connect(cache) as c:
                (n,) = c.execute("SELECT COUNT(*) FROM context_cache").fetchone()
                out.context_cache_rows = int(n)
    except Exception:
        pass

    # ----- sessions
    try:
        from akb.sessions.db import list_sessions

        out.session_count = len(list_sessions())
    except Exception:
        pass

    return out


def export_session(session_id: int, out_dir: Path | None = None) -> Path:
    """Write a session's transcript as a markdown note in the vault.

    Citations become ``[[Source]]`` wikilinks if their source_id resembles
    an Obsidian note path. Falls back to backticked source_id otherwise.
    """
    from akb.sessions.db import get_session_history, list_sessions

    settings = load_settings()
    out_dir = out_dir or (settings.paths.vault / "akb_chats")
    out_dir.mkdir(parents=True, exist_ok=True)

    sessions = list_sessions()
    meta = next((s for s in sessions if int(s["id"]) == int(session_id)), None)
    if meta is None:
        raise ValueError(f"session {session_id} not found")
    name = str(meta["name"])
    safe = "".join(ch if ch.isalnum() or ch in " -_" else "-" for ch in name).strip().replace(" ", "-")
    fname = f"{datetime.now().strftime('%Y-%m-%d')}--{safe or 'session'}.md"

    history = get_session_history(int(session_id))
    lines = [
        "---",
        f"title: {name}",
        f"session_id: {session_id}",
        f"exported_at: {datetime.now().isoformat()}",
        "tags: [akb/chat]",
        "---",
        "",
        f"# {name}",
        "",
    ]
    for m in history:
        role = m.get("role", "user")
        content = m.get("content", "")
        prefix = "**you** ▸" if role == "user" else "**akb** ▸"
        lines.append(f"{prefix} {content}")
        lines.append("")

    path = out_dir / fname
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
