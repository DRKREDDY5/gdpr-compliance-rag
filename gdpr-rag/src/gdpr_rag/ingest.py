"""Download and parse the GDPR text into structured, citable records.

Source: gdpr-info.eu, which publishes the official GDPR text (Regulation (EU)
2016/679) as one clean HTML page per Article and per Recital at predictable
URLs. Fetching many small, well-structured pages is far more robust than
scraping one giant JavaScript-gated EUR-Lex page (which returns an empty body
to direct HTTP clients).

  Articles:  https://gdpr-info.eu/art-{1..99}-gdpr/
  Recitals:  https://gdpr-info.eu/recitals/no-{1..173}/

Run:  python -m src.gdpr_rag.ingest
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict

import requests
from bs4 import BeautifulSoup

from .config import DATA_RAW, DATA_PROCESSED, CHUNKS_PATH

N_ARTICLES = 99
N_RECITALS = 173
BASE = "https://gdpr-info.eu"
HEADERS = {"User-Agent": "gdpr-rag/1.0 (educational portfolio project)"}
REQUEST_PAUSE_SEC = 0.3  # be polite to the source


@dataclass
class Record:
    id: str            # "art-7" / "rec-32"
    kind: str          # "article" | "recital"
    number: str
    title: str
    text: str
    citation: str      # "Art. 7" / "Recital 32"


def _get(url: str) -> str:
    resp = requests.get(url, timeout=30, headers=HEADERS)
    resp.raise_for_status()
    return resp.text


def _clean(text: str) -> str:
    return " ".join(text.split()).strip()


def fetch_article(n: int) -> Record | None:
    """Parse one article page. The main content sits in <div class="entry-content">;
    the heading carries the article number and title."""
    html = _get(f"{BASE}/art-{n}-gdpr/")
    soup = BeautifulSoup(html, "html.parser")

    content = soup.find("div", class_="entry-content")
    title_el = soup.find(["h1", "h2"], class_="entry-title") or soup.find("h1")
    if content is None:
        return None

    title = _clean(title_el.get_text()) if title_el else f"Article {n}"
    # Drop the breadcrumbs/nav noise; keep paragraph + list text.
    body = _clean(content.get_text(" "))
    if len(body) < 30:
        return None

    return Record(
        id=f"art-{n}",
        kind="article",
        number=str(n),
        title=title,
        text=body,
        citation=f"Art. {n}",
    )


def fetch_recital(n: int) -> Record | None:
    html = _get(f"{BASE}/recitals/no-{n}/")
    soup = BeautifulSoup(html, "html.parser")
    content = soup.find("div", class_="entry-content")
    title_el = soup.find(["h1", "h2"], class_="entry-title") or soup.find("h1")
    if content is None:
        return None

    title = _clean(title_el.get_text()) if title_el else f"Recital {n}"
    body = _clean(content.get_text(" "))
    if len(body) < 30:
        return None

    return Record(
        id=f"rec-{n}",
        kind="recital",
        number=str(n),
        title=title,
        text=body,
        citation=f"Recital {n}",
    )


def main() -> None:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    records: list[Record] = []

    print(f"Fetching {N_ARTICLES} articles from gdpr-info.eu ...")
    for n in range(1, N_ARTICLES + 1):
        try:
            rec = fetch_article(n)
            if rec:
                records.append(rec)
            else:
                print(f"  ! Article {n}: no content parsed")
        except Exception as e:
            print(f"  ! Article {n}: {e}")
        time.sleep(REQUEST_PAUSE_SEC)
        if n % 20 == 0:
            print(f"  ... {n}/{N_ARTICLES} articles")

    print(f"Fetching {N_RECITALS} recitals ...")
    for n in range(1, N_RECITALS + 1):
        try:
            rec = fetch_recital(n)
            if rec:
                records.append(rec)
        except Exception as e:
            print(f"  ! Recital {n}: {e}")
        time.sleep(REQUEST_PAUSE_SEC)
        if n % 40 == 0:
            print(f"  ... {n}/{N_RECITALS} recitals")

    if not records:
        raise SystemExit(
            "Parsed 0 records. Inspect a saved page and adjust the selectors "
            "in fetch_article()/fetch_recital()."
        )

    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r)) + "\n")

    n_art = sum(1 for r in records if r.kind == "article")
    n_rec = sum(1 for r in records if r.kind == "recital")
    print(f"\nParsed {len(records)} records ({n_art} articles, {n_rec} recitals).")
    print(f"Wrote {CHUNKS_PATH}")
    print(f"Expected: {N_ARTICLES} articles, {N_RECITALS} recitals.")
    if n_art < N_ARTICLES * 0.9 or n_rec < N_RECITALS * 0.9:
        print("WARNING: counts look low — inspect the pages and selectors.")


if __name__ == "__main__":
    main()