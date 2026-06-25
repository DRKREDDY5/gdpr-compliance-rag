"""LLM completion via Groq — free hosted inference, no local memory needed.

Keeps the laptop doing only lightweight retrieval; the heavy generation runs on
Groq's free hosted endpoint. Swappable: set LLM_MODEL in .env to change models.
"""
from __future__ import annotations

import os
from .config import CONFIG


def complete(system: str, user: str) -> str:
    from groq import Groq
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    resp = client.chat.completions.create(
        model=CONFIG.model.llm_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.0,  # deterministic, grounded answers
    )
    return resp.choices[0].message.content.strip()