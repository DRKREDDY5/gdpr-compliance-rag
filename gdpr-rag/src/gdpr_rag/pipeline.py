"""Convenience entry point tying the whole flow together."""
from __future__ import annotations

from .generate import answer, Answer


def ask(question: str) -> Answer:
    return answer(question)
