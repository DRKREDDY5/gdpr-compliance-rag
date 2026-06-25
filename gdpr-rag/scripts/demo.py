"""Interactive CLI. Usage: python scripts/demo.py "your question" """
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.gdpr_rag.pipeline import ask


def main() -> None:
    if len(sys.argv) > 1:
        questions = [" ".join(sys.argv[1:])]
    else:
        questions = None

    if questions:
        for q in questions:
            _show(q)
    else:
        print("GDPR Q&A — type a question (or 'quit').")
        while True:
            q = input("\n> ").strip()
            if q.lower() in {"quit", "exit", ""}:
                break
            _show(q)


def _show(q: str) -> None:
    res = ask(q)
    print(f"\nQ: {q}")
    print(f"\n{res.text}")
    if res.citations:
        print(f"\nSources: {', '.join(dict.fromkeys(res.citations))}")


if __name__ == "__main__":
    main()
