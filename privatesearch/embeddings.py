"""Embedding back-ends.

* SentenceTransformerEmbedder - real semantic embeddings (all-MiniLM-L6-v2, ~90 MB, CPU friendly).
* HashingEmbedder - a small offline fallback built from hashed words and character n-grams.
  It captures word *form* similarity (agent/agents), NOT meaning. It exists so the app still
  runs when the model cannot be downloaded, and so the automated tests need no network.
"""
from __future__ import annotations

import math
import re
import zlib
from collections import Counter
from typing import Sequence

import numpy as np

from .config import Settings
from .textutil import STOPWORDS, tokenize


class BaseEmbedder:
    name: str = "base"
    dim: int = 0
    is_semantic: bool = False
    min_score: float = 0.25        # minimum cosine for a passage to count as relevant to a query
    link_threshold: float = 0.45   # minimum cosine for a passage-to-passage connection
    fallback_reason: str | None = None

    def encode(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


class HashingEmbedder(BaseEmbedder):
    name = "hashing-ngrams-v1"
    dim = 1024
    is_semantic = False
    min_score = 0.12
    link_threshold = 0.22

    def _features(self, text: str) -> Counter:
        tokens = [t for t in tokenize(text) if t not in STOPWORDS]
        feats: Counter = Counter()
        for tok in tokens:
            feats[f"w:{tok}"] += 1.0
            padded = f"#{tok}#"
            for n in (4, 5):
                for i in range(len(padded) - n + 1):
                    feats[f"c:{padded[i:i + n]}"] += 0.35
        for a, b in zip(tokens, tokens[1:]):
            feats[f"b:{a}_{b}"] += 0.6
        return feats

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feat, weight in self._features(text).items():
                h = zlib.crc32(feat.encode("utf-8"))
                sign = 1.0 if (h >> 16) & 1 else -1.0
                out[row, h % self.dim] += sign * math.log1p(weight)
            norm = np.linalg.norm(out[row])
            if norm > 0:
                out[row] /= norm
        return out


class SentenceTransformerEmbedder(BaseEmbedder):
    is_semantic = True
    min_score = 0.25
    link_threshold = 0.45

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # imported lazily: it is heavy

        self.model = SentenceTransformer(model_name, device="cpu")
        self.model.max_seq_length = min(getattr(self.model, "max_seq_length", 256) or 256, 256)
        self.name = f"st:{model_name}"
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self.model.encode(list(texts), batch_size=32, normalize_embeddings=True,
                                    show_progress_bar=False, convert_to_numpy=True)
        return np.asarray(vectors, dtype=np.float32)


def load_embedder(settings: Settings) -> BaseEmbedder:
    if settings.embedder_backend == "hash":
        return HashingEmbedder()
    try:
        return SentenceTransformerEmbedder(settings.embedding_model)
    except Exception as exc:  # missing package, no model cached and no internet, etc.
        if settings.embedder_backend == "sentence-transformers":
            raise
        fallback = HashingEmbedder()
        fallback.fallback_reason = f"{type(exc).__name__}: {str(exc)[:300]}"
        return fallback
