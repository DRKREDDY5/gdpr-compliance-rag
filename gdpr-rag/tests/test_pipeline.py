"""Unit tests for the pure-logic parts (no network needed).

Run: pytest
These cover the bits that break silently: fusion math, chunk splitting, and the
refusal gate. The retrieval-quality bar is enforced separately by eval/run_eval.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gdpr_rag.retrieve import reciprocal_rank_fusion
from src.gdpr_rag.chunk import split_long


def test_rrf_rewards_agreement():
    # A doc ranked highly by BOTH retrievers should win.
    bm25 = [("a", 0), ("b", 0), ("c", 0)]
    vec = [("b", 0), ("a", 0), ("d", 0)]
    fused = reciprocal_rank_fusion([bm25, vec], rrf_k=60)
    ids = [doc_id for doc_id, _ in fused]
    # "a" (ranks 1,2) and "b" (ranks 2,1) should top "c"/"d" (single list only)
    assert set(ids[:2]) == {"a", "b"}


def test_rrf_handles_disjoint_lists():
    fused = reciprocal_rank_fusion([[("a", 0)], [("b", 0)]], rrf_k=60)
    assert {doc_id for doc_id, _ in fused} == {"a", "b"}


def test_split_long_no_split_when_short():
    assert split_long("short text", max_chars=100, overlap=10) == ["short text"]


def test_split_long_overlaps():
    text = "x" * 250
    parts = split_long(text, max_chars=100, overlap=20)
    assert len(parts) >= 3
    # consecutive parts share the overlap region
    assert parts[0][-20:] == parts[1][:20]


def test_refusal_gate_logic(monkeypatch):
    # If retrieval returns nothing, we must refuse without calling the LLM.
    from src.gdpr_rag import generate

    monkeypatch.setattr(generate, "retrieve", lambda q: [])
    res = generate.answer("anything")
    assert res.refused is True
    assert res.citations == []
