"""Local embedding provider — no API key, no cost, runs offline.

Uses sentence-transformers (already installed for the reranker). The default
model is small, fast, and CPU-friendly. Swap EMBEDDING_MODEL in .env for a
larger one if you want higher quality and have the compute.
"""
from __future__ import annotations

import numpy as np

from .config import CONFIG

_model = None  # lazy singleton so we load the model once


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(CONFIG.model.embedding_model)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a batch of passages (used at index time)."""
    model = _get_model()
    vecs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return np.asarray(vecs, dtype=np.float32)


def embed_query(query: str) -> np.ndarray:
    """Embed a single query (used at search time)."""
    return embed_texts([query])[0]