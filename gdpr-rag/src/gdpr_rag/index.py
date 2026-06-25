"""Build the BM25 and dense vector indexes from parsed chunks.

Run (after ingest):  python -m src.gdpr_rag.index
"""
from __future__ import annotations

import json
import pickle

import numpy as np
from rank_bm25 import BM25Okapi

from .config import CHUNKS_PATH, BM25_PATH, VECTOR_PATH, VECTOR_META_PATH
from .embeddings import embed_texts


def load_chunks() -> list[dict]:
    return [json.loads(l) for l in CHUNKS_PATH.open(encoding="utf-8") if l.strip()]


def build_bm25(chunks: list[dict]) -> None:
    corpus = [c["text"].lower().split() for c in chunks]
    ids = [c["id"] for c in chunks]
    bm25 = BM25Okapi(corpus)
    with BM25_PATH.open("wb") as f:
        pickle.dump((bm25, ids), f)
    print(f"BM25 index: {len(ids)} docs -> {BM25_PATH}")


def build_vectors(chunks: list[dict], batch_size: int = 64) -> None:
    texts = [c["text"] for c in chunks]
    vecs = []
    for i in range(0, len(texts), batch_size):
        vecs.append(embed_texts(texts[i : i + batch_size]))
        print(f"  embedded {min(i + batch_size, len(texts))}/{len(texts)}")
    matrix = np.vstack(vecs)
    np.save(VECTOR_PATH, matrix)
    with VECTOR_META_PATH.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({"id": c["id"]}) + "\n")
    print(f"Vector index: {matrix.shape} -> {VECTOR_PATH}")


def main() -> None:
    chunks = load_chunks()
    if not chunks:
        raise SystemExit("No chunks found. Run `python -m src.gdpr_rag.ingest` first.")
    build_bm25(chunks)
    build_vectors(chunks)
    print("Indexes built.")


if __name__ == "__main__":
    main()
