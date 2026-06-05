"""Cross-session long-term memory.

Every finalized Q/A pair is embedded and stashed in a dedicated Qdrant
collection (``chat_memory``). On each new turn the agent does a soft lookup
against this store and surfaces the top hits as ``[memory]`` blocks alongside
retrieved vault chunks — so the user can ask "what was I working on last
week?" without keeping the whole conversation history in the prompt.

Why a separate collection (vs. tagging memory chunks in the main one):
  * Independent payload schema (``session_id``, ``role``, ``ts``).
  * Different dimensionality / quantization choices are free here.
  * ``akb reindex`` shouldn't nuke conversation memory.

Failure modes are quiet — a broken memory lookup is never a reason to fail
a chat turn.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from akb.config import load_settings
from akb.embed.providers import get_embedder
from akb.obs.logging import get_logger

log = get_logger(__name__)

MEMORY_COLLECTION = "chat_memory"


def _point_id(key: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


@dataclass(frozen=True)
class MemoryHit:
    session_id: int | None
    question: str
    answer: str
    ts: str
    score: float


@dataclass
class MemoryStore:
    """Soft Q/A long-term memory backed by Qdrant.

    Construct lazily — the embedder + Qdrant client are heavy. The CLI / agent
    should grab a singleton via :func:`get_memory`.
    """

    _client: Any = field(default=None, init=False, repr=False)

    def _open(self) -> Any:
        if self._client is None:
            from akb.store.qdrant_store import get_store

            self._client = get_store().client
            self._ensure_collection()
        return self._client

    def _ensure_collection(self) -> None:
        from qdrant_client import models  # type: ignore[import-untyped]

        cl = self._client
        if cl.collection_exists(MEMORY_COLLECTION):
            return
        cfg = load_settings().embed
        cl.create_collection(
            collection_name=MEMORY_COLLECTION,
            vectors_config={
                "dense": models.VectorParams(size=cfg.dim, distance=models.Distance.COSINE)
            },
        )

    # ---------- writes ----------

    def remember(
        self,
        session_id: int | None,
        question: str,
        answer: str,
    ) -> str | None:
        if not (question and answer):
            return None
        try:
            from qdrant_client import models  # type: ignore[import-untyped]

            client = self._open()
            emb = get_embedder().embed_documents([f"Q: {question}\nA: {answer}"])
            ts = datetime.now(timezone.utc).isoformat()
            key = f"sess:{session_id or 0}:{ts}"
            point_id = _point_id(key)
            client.upsert(
                collection_name=MEMORY_COLLECTION,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector={"dense": emb.dense[0]},
                        payload={
                            "session_id": session_id,
                            "question": question,
                            "answer": answer,
                            "ts": ts,
                        },
                    )
                ],
                wait=True,
            )
            log.info("memory.write", session_id=session_id)
            return point_id
        except Exception as e:
            log.warning("memory.write.error", error=str(e))
            return None

    # ---------- reads ----------

    def recall(self, query: str, *, top_k: int = 3) -> list[MemoryHit]:
        if not query:
            return []
        try:
            client = self._open()
            emb = get_embedder().embed_query(query)
            res = client.query_points(
                collection_name=MEMORY_COLLECTION,
                query=emb.dense[0],
                using="dense",
                limit=top_k,
                with_payload=True,
            )
            out: list[MemoryHit] = []
            for sp in res.points:
                p = sp.payload or {}
                out.append(
                    MemoryHit(
                        session_id=p.get("session_id"),
                        question=str(p.get("question", "")),
                        answer=str(p.get("answer", "")),
                        ts=str(p.get("ts", "")),
                        score=float(sp.score),
                    )
                )
            log.info("memory.recall", hits=len(out))
            return out
        except Exception as e:
            log.warning("memory.recall.error", error=str(e))
            return []

    def count(self) -> int:
        try:
            client = self._open()
            return int(client.count(MEMORY_COLLECTION, exact=True).count)
        except Exception:
            return 0

    def clear(self) -> None:
        try:
            client = self._open()
            if client.collection_exists(MEMORY_COLLECTION):
                client.delete_collection(MEMORY_COLLECTION)
        except Exception as e:
            log.warning("memory.clear.error", error=str(e))


def format_memory_block(hits: list[MemoryHit]) -> str:
    if not hits:
        return ""
    parts: list[str] = []
    for h in hits:
        parts.append(
            f"[memory · {h.ts[:10]} · score {h.score:.2f}]\n"
            f"Q: {h.question}\nA: {h.answer}"
        )
    return "\n---\n".join(parts)


_SINGLETON: MemoryStore | None = None
_LOCK = threading.Lock()


def get_memory() -> MemoryStore:
    global _SINGLETON
    if _SINGLETON is None:
        with _LOCK:
            if _SINGLETON is None:
                _SINGLETON = MemoryStore()
    return _SINGLETON


def reset_memory_singleton() -> None:
    global _SINGLETON
    with _LOCK:
        _SINGLETON = None
