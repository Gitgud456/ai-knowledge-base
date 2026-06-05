"""Obsidian vault loader.

Parses markdown notes into ``Document`` objects with:
  - frontmatter (typed via python-frontmatter)
  - tags (frontmatter + inline #tag)
  - wikilinks ``[[Note]]``, ``[[Note|Alias]]``, ``[[Note#Heading]]`` — link targets only
  - aliases (frontmatter ``aliases``)
  - embeds ``![[Note]]`` — expanded inline at parse time, recursive (with cycle guard)

Wikilink resolution is case-insensitive, basename-based (matching Obsidian's "shortest path
when possible" default). Unresolved targets are kept as link text so the retriever can still
match the literal string.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Iterator

import frontmatter

from akb.config import IngestConfig, load_settings
from akb.schemas import Document, SourceType

# [[target]] | [[target|alias]] | [[target#heading]] | [[target#^block]]
WIKILINK_RX = re.compile(r"\[\[([^\]\n]+?)\]\]")
EMBED_RX = re.compile(r"!\[\[([^\]\n]+?)\]\]")
INLINE_TAG_RX = re.compile(r"(?:^|\s)#([A-Za-z0-9_\-/]+)")


def _split_link(raw: str) -> tuple[str, str | None, str | None]:
    """Return (target, heading_or_block, alias) for a wikilink body."""
    alias: str | None = None
    head: str | None = None
    if "|" in raw:
        raw, alias = raw.split("|", 1)
    if "#" in raw:
        raw, head = raw.split("#", 1)
    return raw.strip(), (head.strip() if head else None), (alias.strip() if alias else None)


def _normalise_tags(raw: object) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw.lstrip("#")]
    if isinstance(raw, list):
        out: list[str] = []
        for t in raw:
            if isinstance(t, str):
                out.append(t.lstrip("#"))
        return out
    return []


def _normalise_aliases(raw: object) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return [a for a in raw if isinstance(a, str)]
    return []


def _build_index(vault: Path, cfg: IngestConfig) -> dict[str, Path]:
    """Map normalised target name -> resolved path. Case-insensitive by basename.

    Obsidian's default link mode is "shortest path when possible" — we approximate
    that with basename matching, which covers ~all personal vaults.
    """
    skip = {d.lower() for d in cfg.skip_dirs}
    idx: dict[str, Path] = {}
    for md in vault.rglob("*.md"):
        if any(part.lower() in skip for part in md.parts):
            continue
        idx.setdefault(md.stem.lower(), md)
    return idx


def _read_raw(path: Path) -> tuple[dict[str, object], str]:
    """Return (frontmatter_dict, body). Resilient to malformed YAML."""
    try:
        post = frontmatter.load(path.open("r", encoding="utf-8"))
        return dict(post.metadata or {}), str(post.content)
    except Exception:
        return {}, path.read_text(encoding="utf-8", errors="replace")


def _expand_embeds(
    body: str,
    base: Path,
    vault: Path,
    index: dict[str, Path],
    seen: set[Path],
    depth: int = 0,
    max_depth: int = 3,
) -> str:
    """Recursively inline ``![[Note]]`` embeds. Guards against cycles + depth."""
    if depth >= max_depth:
        return body

    def repl(m: re.Match[str]) -> str:
        target, _head, _alias = _split_link(m.group(1))
        resolved = index.get(target.lower())
        if not resolved or resolved in seen:
            return m.group(0)
        try:
            _, inner = _read_raw(resolved)
        except OSError:
            return m.group(0)
        seen2 = seen | {resolved}
        expanded = _expand_embeds(inner, resolved, vault, index, seen2, depth + 1, max_depth)
        rel = resolved.relative_to(vault) if resolved.is_relative_to(vault) else resolved
        return f"\n\n<!-- embed: {rel} -->\n{expanded}\n<!-- /embed -->\n\n"

    _ = base  # reserved for future per-embed relative resolution
    return EMBED_RX.sub(repl, body)


def _extract_wikilinks(body: str) -> list[str]:
    out: list[str] = []
    for m in WIKILINK_RX.finditer(body):
        target, _, _ = _split_link(m.group(1))
        if target:
            out.append(target)
    # dedupe but keep order
    seen: set[str] = set()
    return [x for x in out if not (x in seen or seen.add(x))]


def _extract_inline_tags(body: str) -> list[str]:
    return list({m.group(1) for m in INLINE_TAG_RX.finditer(body)})


def load_note(path: Path, vault: Path, index: dict[str, Path]) -> Document:
    fm, body = _read_raw(path)
    # Wikilinks are this note's *direct* outbound links — extract from the
    # original body (regex sees `[[X]]` inside `![[X]]` too) so embed targets
    # become graph edges. Content + tags use the post-expansion text.
    wikilinks = _extract_wikilinks(body)
    expanded = _expand_embeds(body, path, vault, index, seen={path})

    tags = sorted(set(_normalise_tags(fm.get("tags"))) | set(_extract_inline_tags(expanded)))
    aliases = _normalise_aliases(fm.get("aliases"))
    title = fm.get("title") if isinstance(fm.get("title"), str) else path.stem
    stat = path.stat()

    rel = path.relative_to(vault) if path.is_relative_to(vault) else path
    source_id = f"obsidian:{rel.as_posix()}"

    return Document(
        source_id=source_id,
        source_path=path,
        source_type=SourceType.obsidian,
        title=title,
        content=expanded,
        frontmatter=fm,
        tags=tags,
        wikilinks=wikilinks,
        aliases=aliases,
        created_at=datetime.fromtimestamp(stat.st_ctime),
        modified_at=datetime.fromtimestamp(stat.st_mtime),
        extra={"relpath": rel.as_posix()},
    )


def iter_vault(vault: Path | None = None, cfg: IngestConfig | None = None) -> Iterator[Document]:
    """Walk the Obsidian vault and yield ``Document`` objects."""
    settings = load_settings()
    vault = vault or settings.paths.vault
    cfg = cfg or settings.ingest
    skip = {d.lower() for d in cfg.skip_dirs}

    index = _build_index(vault, cfg)
    for md in vault.rglob("*.md"):
        if any(part.lower() in skip for part in md.parts):
            continue
        yield load_note(md, vault, index)


def load_vault(vault: Path | None = None) -> list[Document]:
    """Eager wrapper around :func:`iter_vault`. Convenient for small vaults / tests."""
    return list(iter_vault(vault))
