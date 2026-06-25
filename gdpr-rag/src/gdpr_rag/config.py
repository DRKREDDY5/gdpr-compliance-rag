"""Central configuration for the GDPR RAG system.

Every tunable knob lives here so experiments are reproducible and the eval
harness can record exactly what settings produced a given score.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()  # read key/config from the .env file into the environment

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
CHUNKS_PATH = DATA_PROCESSED / "chunks.jsonl"
BM25_PATH = DATA_PROCESSED / "bm25.pkl"
VECTOR_PATH = DATA_PROCESSED / "vectors.npy"
VECTOR_META_PATH = DATA_PROCESSED / "vectors_meta.jsonl"


# ---------------------------------------------------------------------------
# Models  (override via .env)
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    # Embeddings: any provider works; default keeps it swappable.
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    # Cross-encoder reranker (runs locally via sentence-transformers).
    reranker_model: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    # Generation model for the final grounded answer.
    llm_model: str = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    # API key is read by the provider SDK from the environment; never hard-code.
    api_key_env: str = "OPENAI_API_KEY"


# ---------------------------------------------------------------------------
# Retrieval / pipeline parameters
# ---------------------------------------------------------------------------
@dataclass
class RetrievalConfig:
    # How many candidates each retriever returns before fusion.
    bm25_top_k: int = 20
    vector_top_k: int = 20
    # Reciprocal Rank Fusion constant (standard default = 60).
    rrf_k: int = 60
    # How many fused candidates go into the cross-encoder reranker.
    rerank_candidates: int = 15
    # Final passages handed to the generator.
    final_top_k: int = 5
    # Confidence gate: if the top reranker score is below this, REFUSE.
    # Cross-encoder scores are logits; calibrate this on your eval set.
    min_rerank_score: float = -8.0


@dataclass
class ChunkConfig:
    # GDPR is structured by Article/Recital, so we chunk along those
    # boundaries first, then split anything too long.
    max_chars: int = 1200
    overlap_chars: int = 150


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)


CONFIG = Config()
