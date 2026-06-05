"""Image-in-note discovery + ingest.

Walks the vault for ``![[image]]`` and ``![alt](path)`` markdown image embeds,
resolves them against the vault filesystem, and embeds each with SigLIP into
the dedicated ``vault_images`` Qdrant collection.

At query time the agent can do a cross-modal text → image search and surface
hits as a "related image" panel (UI) or as side context (CLI).

Failure modes: any image that fails to open / embed is logged and skipped.
The CLI surfaces the count in the summary.
"""

from __future__ import annotations

import re
import unicodedata
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from akb.config import ImageConfig, load_settings
from akb.embed.multimodal import get_image_embedder
from akb.obs.logging import get_logger
from akb.store.qdrant_store import QdrantStore, get_store

log = get_logger(__name__)


WIKILINK_IMAGE_RX = re.compile(r"!\[\[([^\]\n]+\.(?:png|jpe?g|gif|webp|bmp|svg))(?:\|[^\]]+)?\]\]", re.IGNORECASE)
MD_IMAGE_RX = re.compile(r"!\[[^\]]*\]\(([^)\n]+\.(?:png|jpe?g|gif|webp|bmp|svg))(?:\s+\"[^\"]*\")?\)", re.IGNORECASE)


def _norm(s: str) -> str:
    return unicodedata.normalize("NFC", s.strip()).casefold()


@dataclass(frozen=True)
class ImageRef:
    note_path: Path
    image_path: Path
    raw_target: str


def _vault_image_index(vault: Path, attachment_dirs: list[str]) -> dict[str, Path]:
    """Map normalised basename → absolute path. Searches the configured
    attachment dirs first, then the whole vault as fallback."""
    out: dict[str, Path] = {}
    # First pass — attachment dirs are usually flat
    for sub in attachment_dirs:
        base = vault / sub
        if not base.is_dir():
            continue
        for p in base.rglob("*"):
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg"}:
                out.setdefault(_norm(p.name), p)
    # Whole-vault fallback
    for p in vault.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg"}:
            out.setdefault(_norm(p.name), p)
    return out


def _extract_refs(note_path: Path, vault: Path, index: dict[str, Path]) -> list[ImageRef]:
    text = note_path.read_text(encoding="utf-8", errors="replace")
    out: list[ImageRef] = []
    for m in WIKILINK_IMAGE_RX.finditer(text):
        target = m.group(1).strip()
        resolved = index.get(_norm(Path(target).name))
        if resolved:
            out.append(ImageRef(note_path=note_path, image_path=resolved, raw_target=target))
    for m in MD_IMAGE_RX.finditer(text):
        target = m.group(1).strip()
        cand = (note_path.parent / target).resolve()
        if cand.exists():
            out.append(ImageRef(note_path=note_path, image_path=cand, raw_target=target))
        else:
            resolved = index.get(_norm(Path(target).name))
            if resolved:
                out.append(ImageRef(note_path=note_path, image_path=resolved, raw_target=target))
    return out


def discover_images(
    vault: Path | None = None,
) -> list[ImageRef]:
    """Walk the vault and return every image referenced by a note."""
    settings = load_settings()
    vault = vault or settings.paths.vault
    skip = {_norm(d) for d in settings.ingest.skip_dirs}
    index = _vault_image_index(vault, settings.ingest.attachment_dirs)
    out: list[ImageRef] = []
    seen: set[Path] = set()
    for note in vault.rglob("*.md"):
        if any(_norm(part) in skip for part in note.parts):
            continue
        for ref in _extract_refs(note, vault, index):
            if ref.image_path in seen:
                continue
            seen.add(ref.image_path)
            out.append(ref)
    log.info("images.discover", n=len(out))
    return out


def _ensure_images_collection(store: QdrantStore, cfg: ImageConfig) -> None:
    from qdrant_client import models  # type: ignore[import-untyped]

    cl = store.client
    if cl.collection_exists(cfg.collection):
        return
    cl.create_collection(
        collection_name=cfg.collection,
        vectors_config={
            "dense": models.VectorParams(size=cfg.embed_dim, distance=models.Distance.COSINE)
        },
    )


def _point_id(image_path: Path) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, str(image_path.resolve())))


def ingest_images(
    refs: Iterable[ImageRef] | None = None,
    *,
    store: QdrantStore | None = None,
) -> int:
    """Embed each unique image and upsert into the image collection."""
    settings = load_settings()
    cfg = settings.images
    if not cfg.enabled:
        log.info("images.ingest.skip_disabled")
        return 0
    store = store or get_store()
    _ensure_images_collection(store, cfg)

    if refs is None:
        refs = discover_images()
    refs = list(refs)
    if not refs:
        return 0

    from qdrant_client import models  # type: ignore[import-untyped]

    embedder = get_image_embedder()
    paths = [r.image_path for r in refs]
    embs = embedder.embed_images(paths)
    by_path = {e.path: e.vector for e in embs}

    now = datetime.now(timezone.utc).isoformat()
    points: list[Any] = []
    for ref in refs:
        vec = by_path.get(ref.image_path)
        if not vec:
            continue
        points.append(
            models.PointStruct(
                id=_point_id(ref.image_path),
                vector={"dense": vec},
                payload={
                    "image_path": str(ref.image_path),
                    "note_path": str(ref.note_path),
                    "raw_target": ref.raw_target,
                    "embedded_at": now,
                },
            )
        )
    if not points:
        return 0
    store.client.upsert(collection_name=cfg.collection, points=points, wait=True)
    log.info("images.ingest.done", n=len(points))
    return len(points)


@dataclass
class ImageHit:
    image_path: str
    note_path: str
    score: float


def search_images(query: str, *, top_k: int = 8) -> list[ImageHit]:
    """Cross-modal text → image search."""
    cfg = load_settings().images
    if not cfg.enabled:
        return []
    store = get_store()
    if not store.client.collection_exists(cfg.collection):
        return []
    vec = get_image_embedder().embed_text(query)
    res = store.client.query_points(
        collection_name=cfg.collection,
        query=vec,
        using="dense",
        limit=top_k,
        with_payload=True,
    )
    out: list[ImageHit] = []
    for sp in res.points:
        p = sp.payload or {}
        out.append(
            ImageHit(
                image_path=str(p.get("image_path", "")),
                note_path=str(p.get("note_path", "")),
                score=float(sp.score),
            )
        )
    return out
