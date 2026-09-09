# AGENTS.md

Operational guide for AI coding agents (and humans) working in this repo. Keep changes
small, verify against the real system, and report faithfully.

## What this project is

A retrieval-augmented Q&A system over the public [my.gov.uz](https://my.gov.uz) service
catalog. It answers in Uzbek/Russian, grounded strictly in scraped service pages, and
**refuses rather than guesses**. Hand-rolled Python (no LangChain) over Postgres + pgvector,
with `bge-m3` embeddings and a `bge-reranker-v2-m3` cross-encoder, served behind an
OpenAI-compatible FastAPI app and an Open WebUI frontend. See
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full picture.

## Golden rules

- **Run everything from the repo root.** Data paths (`data/…`) resolve relative to the
  working directory, and `src/` modules import each other by **bare name** (`import rag`),
  so the CWD and `src/` on the path both matter.
- **The code is the source of truth.** If a doc, comment, or memory disagrees with the
  code, trust the code and fix the doc.
- **Never hallucinate a grounded answer.** The two gates (`CLF_OFFTOPIC`, `GATE_FLOOR`) and
  the strict system prompt are the product. Don't loosen them without recalibrating.
- **Don't commit or push unless asked.** When you do, branch off `main` first.

## Layout

| Path | What |
| --- | --- |
| `src/api.py` | OpenAI-compatible FastAPI app (`/v1/models`, `/v1/chat/completions`, `/health`) |
| `src/rag.py` | Embedder, DSN, LLM client (OpenRouter), system prompt, vector retrieval, `build_context` |
| `src/hybrid.py` | Vector + full-text retrieval, RRF fusion, typo-fix, connection pool |
| `src/rerank.py` | Cross-encoder rerank + two-layer gate + follow-up rewrite — **`answer()` is the entrypoint** |
| `src/eval_run.py` | hit@1 / hit@5 / MRR retrieval eval |
| `eval/` | Gate-threshold calibration (`calibrate.py`, `labeled_seed.csv`) |
| `pipeline/` | Offline corpus build (scrape → chunk+synonyms → embed → vocab) |
| `data/` | Shipped corpus + labeled eval set |
| `tests/` | pytest critical-path suite (LLM mocked; DB/GPU tests skip cleanly) |
| `topic_clf.joblib` | Trained on/off-topic classifier |

## Setup

```bash
# 1. Postgres + pgvector (matches the default DSN)
docker run -d --name mygov-pg -e POSTGRES_PASSWORD=mygov -e POSTGRES_DB=mygov \
  -p 5433:5432 pgvector/pgvector:pg16

# 2. Python deps
python -m venv mygov && source mygov/bin/activate
pip install -r requirements.txt

# 3. API key
cp .env.example .env   # set OPENROUTER_API_KEY, and PIN OPENROUTER_MODEL (see below)

# 4. Build the index (repo ships data/chunks.jsonl, so no scrape needed)
python pipeline/embed_store.py    # embed chunks → pgvector (TRUNCATEs + reloads)
python pipeline/build_vocab.py    # trigram vocab table for typo-fix
```

## Run

```bash
# Backend API (bind 0.0.0.0 so the Open WebUI container can reach it)
uvicorn api:app --app-dir src --host 0.0.0.0 --port 8000

# Frontend
docker compose up -d      # Open WebUI
```

Smoke-test the API directly:

```bash
curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"mygov-rag","messages":[{"role":"user","content":"tonirovka ruxsatnomasi qanday olinadi?"}]}'
```

## Test & verify

```bash
pytest -v                              # critical-path suite (LLM mocked)
CUDA_VISIBLE_DEVICES="" pytest -v      # if sharing a GPU with the running API (avoids OOM)
python src/eval_run.py                 # retrieval metrics on the labeled set
```

- **Changed retrieval or a gate?** Run `src/eval_run.py` and, for thresholds,
  `eval/calibrate.py eval/labeled_seed.csv` (no answer-LLM, cheap).
- **Changed the answer path?** Smoke-test with an on-topic query (expect a citation), an
  off-topic query (expect a refusal, no sources), and a two-turn follow-up (expect the
  second turn to resolve against the first).

## Conventions

- Match the surrounding style: module docstring with a one-line run example, and comments
  that explain **why**, not what.
- Config comes from env vars read in `src/rag.py` — don't hard-code secrets or hosts.
- Tests must **skip with a reason** when Postgres or the GPU models are unavailable, never
  fake a pass.
- Keep the two gates and the refusal contract intact; both refusal reasons
  (`OUT_OF_SCOPE`, `NO_SUPPORTING_CONTEXT`) must return empty `sources`.

## Gotchas

- **`uvicorn --app-dir src` for offline models.** Some import styles fail to resolve the
  offline Hugging Face cache; run from the repo root as shown.
- **Pin `OPENROUTER_MODEL`.** The default `openrouter/free` auto-router is flaky across
  calls; pin a real model (e.g. `openai/gpt-4o-mini`) for reliable multi-turn.
- **First run downloads ~2.2 GB × 2** (bge-m3, bge-reranker-v2-m3), then caches.
- **Follow-up rewrite is anaphora-gated.** Only anaphoric/short questions are rewritten, so
  a new self-contained question doesn't drag in the previous topic. History is likewise fed
  to the answer LLM **only** for genuine follow-ups.
