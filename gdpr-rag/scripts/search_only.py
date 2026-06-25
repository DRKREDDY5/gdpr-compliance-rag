"""Retrieval-only demo — shows what the system retrieves WITHOUT needing an LLM.

Use this to verify hybrid retrieval + reranking works before setting up Ollama
for answer generation.

Usage:  python scripts/search_only.py "your question"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.gdpr_rag.retrieve import retrieve


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python scripts/search_only.py "your question"')
        return
    question = " ".join(sys.argv[1:])
    hits = retrieve(question)
    print(f"\nQuestion: {question}\n")
    print(f"Top {len(hits)} retrieved passages:\n")
    for i, h in enumerate(hits, 1):
        snippet = h.text[:200].replace("\n", " ")
        print(f"{i}. [{h.citation}]  (rerank score: {h.score:.3f})")
        print(f"   {snippet}...\n")


if __name__ == "__main__":
    main()