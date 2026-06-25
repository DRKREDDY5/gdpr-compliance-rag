"""Grounded answer generation with mandatory citations and refusal-on-uncertainty.

Two production-critical behaviors live here:
  1. The model is instructed to answer ONLY from retrieved passages and to cite
     the Article/Recital for every claim.
  2. If first-stage confidence is too low, we refuse BEFORE calling the LLM —
     cheaper, and it removes the temptation for the model to confabulate.
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import CONFIG
from .retrieve import retrieve, Hit

REFUSAL_MESSAGE = (
    "I couldn't find a confident answer to that in the GDPR text I have indexed. "
    "I won't guess on a compliance question. Try rephrasing, or consult the "
    "regulation directly / a qualified data-protection professional."
)

SYSTEM_PROMPT = """You are a careful assistant that answers questions about the \
EU General Data Protection Regulation (GDPR) using ONLY the provided passages.

Rules:
- Use only the passages below. Do not rely on outside knowledge.
- Cite the source for every claim using its label in square brackets, e.g. [Art. 7] or [Recital 32].
- If the passages do not contain the answer, say you don't have it. Do not guess.
- Be precise and concise. This is informational, not legal advice.
"""


@dataclass
class Answer:
    text: str
    citations: list[str]
    refused: bool
    hits: list[Hit]


def _build_context(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{h.citation}]\n{h.text}" for h in hits)


def answer(question: str) -> Answer:
    hits = retrieve(question)

    # Confidence gate: refuse before spending an LLM call.
    if not hits or hits[0].score < CONFIG.retrieval.min_rerank_score:
        return Answer(text=REFUSAL_MESSAGE, citations=[], refused=True, hits=hits)

    from .llm import complete  # provider-specific; isolated for testing

    context = _build_context(hits)
    user_prompt = f"Passages:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    response = complete(system=SYSTEM_PROMPT, user=user_prompt)

    citations = [h.citation for h in hits if h.citation in response]
    return Answer(text=response, citations=citations, refused=False, hits=hits)
