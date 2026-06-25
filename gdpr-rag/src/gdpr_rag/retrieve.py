"""Hybrid retrieval: BM25 + dense vectors, fused with RRF, then reranked.

Why hybrid? Legal text is full of exact tokens ("Article 17", "data
portability", "controller") where keyword search (BM25) is strong, plus
paraphrased concepts where dense embeddings win. Fusing both and then
reranking with a cross-encoder beats any single retriever — this is the core
"production RAG" upgrade over a naive vector-only demo.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass

import numpy as np

from .config import CONFIG, CHUNKS_PATH, BM25_PATH, VECTOR_PATH, VECTOR_META_PATH


@dataclass
class Hit:
    id: str
    text: str
    citation: str
    score: float


def _load_chunks() -> dict[str, dict]:
    chunks = {}
    with CHUNKS_PATH.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            chunks[rec["id"]] = rec
    return chunks


def bm25_search(query: str, top_k: int) -> list[tuple[str, float]]:
    """Keyword retrieval via a prebuilt BM25 index (see index.py)."""
    with BM25_PATH.open("rb") as f:
        bm25, ids = pickle.load(f)
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    ranked = np.argsort(scores)[::-1][:top_k]
    return [(ids[i], float(scores[i])) for i in ranked]


def vector_search(query: str, top_k: int) -> list[tuple[str, float]]:
    """Dense retrieval via cosine similarity over prebuilt embeddings."""
    from .embeddings import embed_query  # provider-specific; isolated for testing

    vectors = np.load(VECTOR_PATH)
    ids = [json.loads(l)["id"] for l in VECTOR_META_PATH.open(encoding="utf-8")]
    q = embed_query(query)
    q = q / (np.linalg.norm(q) + 1e-8)
    mat = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
    sims = mat @ q
    ranked = np.argsort(sims)[::-1][:top_k]
    return [(ids[i], float(sims[i])) for i in ranked]


def reciprocal_rank_fusion(
    rankings: list[list[tuple[str, float]]], rrf_k: int
) -> list[tuple[str, float]]:
    """Combine multiple ranked lists into one. Rank-based, so the differing
    score scales of BM25 vs cosine don't need normalizing."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, (doc_id, _score) in enumerate(ranking):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


_reranker = None  # lazy singleton: load the cross-encoder once, not per query


def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(CONFIG.model.reranker_model)
    return _reranker


def rerank(query: str, candidate_ids: list[str], chunks: dict) -> list[Hit]:
    """Cross-encoder reranking: scores (query, passage) jointly, which is more
    accurate than the bi-encoder similarity used for first-stage retrieval."""
    model = _get_reranker()
    pairs = [[query, chunks[cid]["text"]] for cid in candidate_ids]
    scores = model.predict(pairs)
    hits = [
        Hit(
            id=cid,
            text=chunks[cid]["text"],
            citation=chunks[cid]["citation"],
            score=float(s),
        )
        for cid, s in zip(candidate_ids, scores)
    ]
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits


def retrieve(query: str) -> list[Hit]:
    """Full retrieval stack → reranked hits (not yet confidence-gated)."""
    cfg = CONFIG.retrieval
    chunks = _load_chunks()

    bm25_hits = bm25_search(query, cfg.bm25_top_k)
    vec_hits = vector_search(query, cfg.vector_top_k)

    fused = reciprocal_rank_fusion([bm25_hits, vec_hits], cfg.rrf_k)
    candidate_ids = [doc_id for doc_id, _ in fused[: cfg.rerank_candidates]]

    reranked = rerank(query, candidate_ids, chunks)
    return reranked[: cfg.final_top_k]