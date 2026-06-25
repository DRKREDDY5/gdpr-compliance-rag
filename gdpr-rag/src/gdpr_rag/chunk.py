"""Article/Recital-aware chunking.

Most GDPR articles fit in one chunk. A few long ones exceed the limit and get
split with overlap so no sentence is orphaned across a boundary. Source
metadata (citation) is preserved on every sub-chunk so citations survive
splitting.
"""
from __future__ import annotations

from .config import CONFIG


def split_long(text: str, max_chars: int, overlap: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    parts, start = [], 0
    while start < len(text):
        end = start + max_chars
        parts.append(text[start:end])
        start = end - overlap
    return parts


def chunk_record(record: dict) -> list[dict]:
    cfg = CONFIG.chunk
    pieces = split_long(record["text"], cfg.max_chars, cfg.overlap_chars)
    if len(pieces) == 1:
        return [record]
    out = []
    for i, piece in enumerate(pieces):
        out.append({**record, "id": f"{record['id']}-p{i}", "text": piece})
    return out
