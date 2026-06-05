"""SigLIP joint text-image embedder.

Used for image-in-note search. Both image and query text get embedded into
the same vector space, so a user query like "the diagram with the dotted
arrows" can hit a screenshot in their vault.

We use the HuggingFace ``transformers`` integration (no extra deps beyond
what's already required for sentence-transformers). The model is downloaded
on first use and cached locally — opt-in via ``images.enabled``.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from akb.config import ImageConfig, load_settings
from akb.obs.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class ImageEmbedding:
    path: Path
    vector: list[float]


class SigLIPEmbedder:
    def __init__(self, cfg: ImageConfig) -> None:
        self._cfg = cfg
        self._model = None
        self._processor = None
        self._lock = threading.Lock()

    @property
    def dim(self) -> int:
        return self._cfg.embed_dim

    def _load(self) -> tuple[Any, Any]:
        if self._model is None or self._processor is None:
            with self._lock:
                if self._model is None or self._processor is None:
                    from transformers import AutoModel, AutoProcessor  # type: ignore[import-untyped]

                    log.info("siglip.load", model=self._cfg.model)
                    self._processor = AutoProcessor.from_pretrained(self._cfg.model)
                    self._model = AutoModel.from_pretrained(self._cfg.model)
                    self._model.eval()
        return self._model, self._processor

    def embed_images(self, paths: list[Path]) -> list[ImageEmbedding]:
        if not paths:
            return []
        from PIL import Image  # pillow ships with streamlit
        import torch

        model, processor = self._load()
        out: list[ImageEmbedding] = []
        for i in range(0, len(paths), self._cfg.batch_size):
            batch = paths[i : i + self._cfg.batch_size]
            images = []
            kept: list[Path] = []
            for p in batch:
                try:
                    images.append(Image.open(p).convert("RGB"))
                    kept.append(p)
                except Exception as e:
                    log.warning("siglip.open.error", path=str(p), error=str(e))
            if not images:
                continue
            inputs = processor(images=images, return_tensors="pt")
            with torch.no_grad():
                features = model.get_image_features(**inputs)
                features = features / features.norm(dim=-1, keepdim=True)
            vecs = features.cpu().tolist()
            for p, v in zip(kept, vecs):
                out.append(ImageEmbedding(path=p, vector=v))
        return out

    def embed_text(self, text: str) -> list[float]:
        import torch

        model, processor = self._load()
        inputs = processor(
            text=[text], return_tensors="pt", padding="max_length", truncation=True
        )
        with torch.no_grad():
            features = model.get_text_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        return features[0].cpu().tolist()


@lru_cache(maxsize=1)
def get_image_embedder() -> SigLIPEmbedder:
    return SigLIPEmbedder(load_settings().images)
