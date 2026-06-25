"""Evaluation harness. Computes retrieval + answer-quality metrics against a
labeled set and EXITS NON-ZERO if any metric falls below threshold.

Prints progress per question and always reports a results table, even if an
individual question errors out (it's counted as a miss, not a crash).

Run:  python eval/run_eval.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.gdpr_rag.retrieve import retrieve          # noqa: E402
from src.gdpr_rag.generate import answer            # noqa: E402

EVAL_SET = Path(__file__).parent / "eval_set.jsonl"
THRESHOLDS = Path(__file__).parent / "thresholds.yaml"
RECALL_K = 5


def load_cases() -> list[dict]:
    return [json.loads(l) for l in EVAL_SET.open(encoding="utf-8") if l.strip()]


def evaluate() -> dict[str, float]:
    cases = load_cases()
    answerable = [c for c in cases if not c["should_refuse"]]
    refusable = [c for c in cases if c["should_refuse"]]

    recall_hits = 0
    reciprocal_ranks = []
    citation_covered = 0
    refusal_correct = 0

    print(f"\nEvaluating {len(answerable)} answerable + {len(refusable)} refusal cases...\n")

    for i, c in enumerate(answerable, 1):
        q = c["question"]
        print(f"  [{i}/{len(answerable)}] {q[:55]}...")
        try:
            hits = retrieve(q)
            found = [h.citation for h in hits[:RECALL_K]]
            expected = set(c["expected_sources"])

            if expected & set(found):
                recall_hits += 1
                for rank, cit in enumerate(found, start=1):
                    if cit in expected:
                        reciprocal_ranks.append(1.0 / rank)
                        break
            else:
                reciprocal_ranks.append(0.0)

            ans = answer(q)
            if ans.citations:
                citation_covered += 1
        except Exception as e:
            print(f"      ! error: {e}")
            reciprocal_ranks.append(0.0)

    print(f"\n  Checking {len(refusable)} out-of-corpus (should refuse) cases...")
    for c in refusable:
        try:
            ans = answer(c["question"])
            if ans.refused:
                refusal_correct += 1
        except Exception as e:
            print(f"      ! error: {e}")

    n_ans = max(len(answerable), 1)
    n_ref = max(len(refusable), 1)

    return {
        "recall_at_k": recall_hits / n_ans,
        "mrr": sum(reciprocal_ranks) / n_ans,
        "citation_coverage": citation_covered / n_ans,
        "refusal_correctness": refusal_correct / n_ref,
    }


def main() -> None:
    thresholds = yaml.safe_load(THRESHOLDS.read_text())
    metrics = evaluate()

    print("\n=== GDPR RAG Evaluation ===")
    failed = []
    for name, value in metrics.items():
        floor = thresholds.get(name, 0.0)
        ok = value >= floor
        flag = "PASS" if ok else "FAIL"
        print(f"  {name:22s} {value:6.3f}  (min {floor:.2f})  [{flag}]")
        if not ok:
            failed.append(name)

    if failed:
        print(f"\nFAILED: {', '.join(failed)} below threshold.")
        sys.exit(1)
    print("\nAll metrics passed.")


if __name__ == "__main__":
    main()