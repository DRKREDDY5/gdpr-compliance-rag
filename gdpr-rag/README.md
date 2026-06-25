# GDPR Compliance Q&A — Production RAG System

A retrieval-augmented question-answering system over the **full official text of the EU General Data Protection Regulation (GDPR)**. Ask a question like *"What are the conditions for valid consent?"* and get an answer grounded in — and citing — the specific GDPR Articles and Recitals it came from. If the answer isn't in the regulation, the system says so instead of guessing.

This is **Project 1 of 5** in a production-AI-engineering portfolio. It is deliberately built to close the gap between a RAG *demo* and a RAG system that could actually run in production:

- **Source grounding + citations** — every answer cites the exact Article/Recital. No citation, no claim.
- **Hybrid retrieval** — BM25 (keyword) + dense vector search, fused with Reciprocal Rank Fusion.
- **Cross-encoder reranking** — a reranker reorders candidates so the most relevant passages win.
- **Refusal on low confidence** — if retrieval doesn't clear a relevance threshold, the system declines rather than hallucinating legal advice.
- **Evaluation harness + CI gate** — a labeled eval set runs in GitHub Actions and **fails the build** if retrieval or answer quality drops below threshold.

> ⚠️ **Not legal advice.** This is an engineering portfolio project that retrieves and summarizes regulatory text. It is not a substitute for a qualified data-protection lawyer.

---

## Why this project

The gap between a RAG demo and production-ready RAG is enormous — and that gap is where the engineering actually lives. A demo retrieves some chunks and pastes them into a prompt. A production system has to answer:

- How do I know retrieval is any good? (→ eval set + metrics)
- What happens when the answer *isn't* in the corpus? (→ refusal logic)
- How do I stop quality silently regressing when I change a prompt or a chunk size? (→ CI gate)
- Why hybrid retrieval instead of just vector search? (→ legal text is full of exact terms — "Article 17", "data portability" — where keyword search beats embeddings; the combination beats either alone)

GDPR is a good corpus for this because it is **public, authoritative, highly structured** (99 Articles + 173 Recitals, each individually numbered and citable), and **high-stakes** — which is exactly what makes grounded citations and refusal-on-uncertainty matter instead of being decoration.

---

## Architecture

```
                        ┌─────────────────────────────────────────────┐
   User question  ──▶   │  1. RETRIEVE                                 │
                        │     ├─ BM25 keyword search                   │
                        │     └─ Dense vector search (embeddings)      │
                        │           │                                  │
                        │           ▼                                  │
                        │     Reciprocal Rank Fusion (combine both)    │
                        │           │                                  │
                        │           ▼                                  │
                        │  2. RERANK                                   │
                        │     Cross-encoder reorders top-N candidates  │
                        │           │                                  │
                        │           ▼                                  │
                        │  3. CONFIDENCE GATE                          │
                        │     top score < threshold ──▶ REFUSE         │
                        │           │ (passes)                         │
                        │           ▼                                  │
                        │  4. GENERATE                                 │
                        │     LLM answers using ONLY retrieved         │
                        │     passages, must cite Article/Recital      │
                        └─────────────────────────────────────────────┘
                                    │
                                    ▼
                        Answer + inline citations (e.g. "[Art. 7(1)]")
```

Each ingested chunk keeps its **source metadata** (Article number, Recital number, title) so a citation can be attached to every retrieved passage and surfaced in the final answer.

---

## Project structure

```
gdpr-rag/
├── README.md
├── requirements.txt
├── .env.example                  # copy to .env, add your API key
├── src/gdpr_rag/
│   ├── ingest.py                 # download + parse GDPR into structured chunks
│   ├── chunk.py                  # article/recital-aware chunking
│   ├── index.py                  # build BM25 + vector indexes
│   ├── retrieve.py               # hybrid retrieval + RRF + reranking
│   ├── generate.py               # grounded answer generation + refusal
│   ├── pipeline.py               # ties retrieve → gate → generate together
│   └── config.py                 # all tunable knobs in one place
├── eval/
│   ├── eval_set.jsonl            # labeled questions w/ expected source articles
│   ├── run_eval.py               # computes retrieval + answer metrics
│   └── thresholds.yaml           # the bar CI enforces
├── scripts/
│   └── demo.py                   # quick CLI to ask questions interactively
├── tests/
│   └── test_pipeline.py          # unit tests for chunking, fusion, gating
└── .github/workflows/eval.yml    # CI: runs eval, fails build if below threshold
```

---

## Setup

> Requires Python 3.10+. You'll run this locally — it needs network access (to download the GDPR text and call an embedding/LLM provider) that the build environment doesn't have.

```bash
# 1. Clone and enter
git clone <your-repo-url> && cd gdpr-rag

# 2. Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Configure your model provider
cp .env.example .env
#   then edit .env and add your API key

# 4. Download + index the GDPR corpus (one time)
python -m src.gdpr_rag.ingest      # fetches official text, parses to chunks
python -m src.gdpr_rag.index       # builds BM25 + vector indexes

# 5. Ask a question
python scripts/demo.py "What are the conditions for valid consent?"

# 6. Run the evaluation
python eval/run_eval.py            # prints metrics, exits non-zero if below threshold
```

---

## Evaluation

The eval set (`eval/eval_set.jsonl`) is a list of questions, each labeled with the GDPR Article(s) that *should* be retrieved to answer it. `run_eval.py` measures:

| Metric | What it tells you |
|---|---|
| **Recall@k** | did we retrieve the right article in the top-k? |
| **MRR** | how high up was the right article ranked? |
| **Citation coverage** | what % of answers cited a source? |
| **Refusal correctness** | did it correctly refuse on out-of-corpus questions? |

`eval/thresholds.yaml` sets the minimum bar. The CI workflow runs the eval on every push and **fails if any metric drops below its threshold** — so you can't merge a change that silently makes retrieval worse. This is the "version your prompts like code, gate deployments on evaluation metrics" discipline from the production-AI playbook.

---

## What I'd build next (production hardening)

Honest list of what separates this from a system you'd actually deploy — good interview talking points:

- **Observability** — per-stage tracing (retrieval latency, rerank latency, token cost per query), which is Project 3 in this series.
- **Caching** — embed-once, cache frequent queries.
- **Eval set expansion** — current set is hand-labeled and small; production needs hundreds of cases + adversarial/out-of-scope questions.
- **Multi-jurisdiction** — UK GDPR, CCPA: the refusal logic matters even more when corpora overlap but differ.

---

## License & attribution

GDPR text is official EU legislation (Regulation (EU) 2016/679), published via EUR-Lex and reusable. This project's code is MIT licensed.
