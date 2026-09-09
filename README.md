# my.gov.uz RAG — Uzbek Government-Services Q&A

> Grounded, citation-backed answers about Uzbek government services — in Uzbek and Russian —
> that refuse rather than guess.

![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![Postgres + pgvector](https://img.shields.io/badge/store-Postgres%20%2B%20pgvector-336791?logo=postgresql&logoColor=white)
![Embeddings](https://img.shields.io/badge/embeddings-bge--m3-6E56CF)
![Reranker](https://img.shields.io/badge/reranker-bge--reranker--v2--m3-6E56CF)
![API](https://img.shields.io/badge/API-OpenAI--compatible-000000)
![License](https://img.shields.io/badge/use-research%20%26%20education-informational)

A retrieval-augmented question-answering system over the public service catalog of
[my.gov.uz](https://my.gov.uz). It answers questions in **Uzbek and Russian** about how to
obtain government services — required documents, fees, deadlines, and steps — grounded
strictly in the scraped service pages, and **refuses rather than guesses** when a question
is off-topic or unsupported by the corpus.

Everything is hand-rolled Python — no LangChain, no vector-DB SaaS — running locally
against **Postgres + pgvector**, with **`bge-m3`** for embeddings and
**`bge-reranker-v2-m3`** for reranking. It ships as an **OpenAI-compatible HTTP API**,
fronted by an [Open WebUI](https://github.com/open-webui/open-webui) chat interface that
talks to the API over its `/v1` endpoints.

**Contents:** [Demo](#demo) · [Highlights](#highlights) · [Architecture](#architecture) ·
[Quickstart](#quickstart) · [Results](#results) · [Configuration](#configuration) ·
[Project structure](#project-structure) · [Tests](#tests) ·
[Refusals & gate calibration](#refusals--gate-calibration) ·
[Documentation](#documentation)

## Demo

Real conversations in the Open WebUI frontend — grounded answers with citations, multi-turn
context, and reason-aware refusals, all in Uzbek.

**Grounded answer + out-of-scope refusal.** A car-history question is answered from service
**/621** (fee, steps, documents); an off-topic question — *"who founded Telegram?"* — is
politely refused instead of answered:

![Grounded answer and an out-of-scope refusal](docs/screenshots/01-grounded-and-refusal.png)

**Multi-service answer + context-carrying follow-up.** One query returns every related
tinting-permit service with its price tiers and citations; the follow-up *"how many people
used these?"* is resolved against the previous turn, not re-guessed:

![Multi-service answer and a context follow-up](docs/screenshots/02-multi-service-and-context.png)

**Broad multi-service listing.** A single topic query — military-service–related services —
returns the full set, each with its own citation:

![Multi-service listing with citations](docs/screenshots/03-multi-service.png)

---

## Highlights

- **Grounded or silent.** A two-layer gate refuses off-topic or weakly-supported questions
  *before* the LLM is ever called — no hallucinated fees or deadlines. On the eval set it
  refuses **6/6** off-topic questions.
- **Hybrid retrieval.** Vector search (meaning) and Postgres full-text search (exact words)
  are fused with Reciprocal Rank Fusion, then re-scored by a cross-encoder. The correct
  service lands in the **top 5 on 22/22** labeled questions.
- **Typo-tolerant.** Out-of-vocabulary query tokens are spell-corrected against a trigram
  index of the real corpus vocabulary, so *ikadastr* still finds *kadastr*.
- **Multilingual & loanword-aware.** A hand-curated synonym map makes the loanwords people
  actually type — *tonirovka*, *propiska* — findable even when the official page uses
  different wording.
- **Conversational.** Follow-ups like *"narxi qancha?"* are rewritten into standalone
  queries using recent history, so retrieval stays accurate across turns.
- **Drop-in UI.** The OpenAI-compatible API works with Open WebUI out of the box, or any
  OpenAI client.

---

## Architecture

```mermaid
flowchart TD
    subgraph build["Offline — corpus build (pipeline/)"]
        S[scrape_services.py<br/>my.gov.uz pages] --> C[chunk_with_synonyms.py<br/>~500-token chunks + synonyms]
        C --> E[embed_store.py<br/>bge-m3 → pgvector]
        C --> V[build_vocab.py<br/>trigram vocab for typo-fix]
    end

    subgraph serve["Online — request path (src/)"]
        Q[User question] --> RW[rewrite_query<br/>follow-up → standalone]
        RW --> G1{Classifier gate<br/>P on-topic ≥ 0.20?}
        G1 -- no --> R[Refuse:<br/>&quot;Menda bu haqda<br/>ishonchli ma'lumot yo'q.&quot;]
        G1 -- yes --> H[Hybrid retrieve<br/>vector + FTS → RRF]
        H --> RR[Cross-encoder rerank<br/>bge-reranker-v2-m3]
        RR --> G2{Grounding floor<br/>top score ≥ 0.05?}
        G2 -- no --> R
        G2 -- yes --> L[LLM answer<br/>OpenRouter + strict prompt]
        L --> A[Grounded answer<br/>+ service citations]
    end

    E -.-> H
    V -.-> H
```

**Corpus build (offline).** The public, server-rendered service pages are scraped into one
JSON record per service (`pipeline/scrape_services.py`). Records are split into ~500-token
chunks that break on sentence boundaries with a small overlap
(`pipeline/chunk_with_synonyms.py`); the same step injects a hand-curated synonym map so
searchable loanwords become findable. Each chunk is embedded with `bge-m3` (1024-dim,
cosine) and stored in Postgres/pgvector alongside its title, URL, and keywords
(`pipeline/embed_store.py`). A trigram vocabulary table for typo-correction is built from
the corpus (`pipeline/build_vocab.py`).

**Retrieval (online).** A query is answered by two independent searches over the chunk
table — a vector nearest-neighbour search that matches on *meaning*, and a Postgres
full-text search that matches on *exact words* — whose ranked lists are fused with
Reciprocal Rank Fusion (`src/hybrid.py`). Before the full-text search runs, every
out-of-vocabulary query token is spell-corrected against the trigram vocabulary. The fused
candidate pool is then re-scored by the `bge-reranker-v2-m3` cross-encoder, which reads
each (query, chunk) pair together for a sharper ordering than either first-stage signal
alone (`src/rerank.py`).

**Grounding gate (what makes answers trustworthy).** Two layers sit in front of the LLM.
First, a logistic-regression classifier over the query embedding estimates the probability
the question is even about government services; below a threshold the system refuses
immediately — without retrieving or calling the LLM — which is what catches *"what's the
weather today?"*. Second, if retrieval runs but the best reranked chunk scores below a
grounding floor, the system refuses rather than answer from weak context. Only when both
gates pass is the top context handed to the LLM (via OpenRouter) under a strict system
prompt: answer *only* from the provided context, always cite the service name and URL, and
emit the fixed refusal sentence when the context lacks the answer.

**Follow-ups.** A single step sits *in front* of the gate, leaving the grounding logic
untouched: a cheap LLM call rewrites the latest message into a standalone query using
recent conversation, so *"narxi qancha?"* after a kadastr-passport question becomes
*"turar-joy kadastr pasporti narxi qancha?"* (`rewrite_query` in `src/rerank.py`). The two
context needs are then met separately — **retrieval and both gates run on the rewritten
standalone query** (right chunks), while the **final answer call also receives the prior
turns** (natural pronouns and tone). Sessions live in the UI layer: Open WebUI keeps each
conversation and sends the full `messages` array every request, so no server-side session
store is needed. The rewrite runs at `temperature=0` and degrades to the raw question on
any error, so a rewriter hiccup falls back to single-turn behaviour rather than crashing.

---

## Quickstart

Everything runs on one machine. A CUDA GPU is recommended (the reranker is a
cross-encoder); CPU works but is slower. The two models download (~2.2 GB each) on first
run and are cached afterward. **Run every command from the repo root** — data paths resolve
relative to the working directory, and the `src/` modules import each other by bare name.

### 1. Start Postgres + pgvector

```bash
docker run -d --name mygov-pg \
  -e POSTGRES_PASSWORD=mygov -e POSTGRES_DB=mygov \
  -p 5433:5432 pgvector/pgvector:pg16
```

This matches the default DSN in the code: `postgresql://postgres:mygov@localhost:5433/mygov`.

### 2. Install dependencies

```bash
python -m venv mygov && source mygov/bin/activate
pip install -r requirements.txt
```

> **torch** is a large, platform-specific wheel. If the plain install picks the wrong
> build for your CUDA/CPU, install it first from the official index, e.g.
> `pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu121`.

### 3. Add your API key

```bash
cp .env.example .env
# edit .env → OPENROUTER_API_KEY=...
```

Get a key at [openrouter.ai/keys](https://openrouter.ai/keys). See
[Configuration](#configuration) for the optional variables — in particular pin
`OPENROUTER_MODEL` for reliable multi-turn.

### 4. Build the index

The repo ships the corpus (`data/chunks.jsonl`), so you can skip scraping. Embed the chunks
into Postgres, then build the typo-fix vocabulary:

```bash
python pipeline/embed_store.py     # embed 926 chunks → pgvector (TRUNCATEs + reloads)
python pipeline/build_vocab.py     # build the trigram vocab table for typo-fix
```

<details>
<summary>Optional: rebuild the corpus from scratch instead of using the shipped data</summary>

```bash
python pipeline/scrape_services.py                              # → data/services.jsonl
python pipeline/chunk_with_synonyms.py \
    data/services.jsonl data/synonyms.jsonl data/chunks.jsonl   # → data/chunks.jsonl
```
</details>

### 5. Start the API

OpenAI-compatible; this is the backend the chat UI talks to. Bind to `0.0.0.0` (not the
default `127.0.0.1`) so the Open WebUI container can reach it via `host.docker.internal`:

```bash
uvicorn api:app --app-dir src --host 0.0.0.0 --port 8000
```

It exposes `GET /v1/models`, `POST /v1/chat/completions`, and `GET /health`. Hit it
directly to verify:

```bash
curl -s localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"mygov-rag","messages":[{"role":"user","content":"tonirovka ruxsatnomasi qanday olinadi?"}]}'
```

On-topic queries return a grounded answer plus the services that grounded it (in a
non-standard `sources` field). Queries that can't be answered get a short, reason-aware
refusal that acknowledges the question **without answering it** — the two failure modes
are kept apart: **out-of-scope** (the classifier judges it isn't a my.gov.uz topic, e.g.
*"bugun ob-havo qanday?"*) vs. **no supporting context** (plausibly on-topic, but retrieval
found nothing that grounds an answer). Each refusal is written by a guarded fallback LLM
call that degrades to a fixed per-reason sentence (e.g. `Menda bu haqda ishonchli ma'lumot
yo'q.`) if the model is unavailable, so the refusal path never errors or hallucinates.

### 6. Start the Open WebUI chat frontend

```bash
docker compose up -d      # then open http://localhost:3000
```

On first visit, Open WebUI asks you to create a **local** admin account (stored in a Docker
volume, not sent anywhere). The `mygov-rag` model then appears in the picker automatically.
The connection is wired in `docker-compose.yml`
(`OPENAI_API_BASE_URL=http://host.docker.internal:8000/v1`); if the API runs on a different
host or port, edit that value.

<details>
<summary>No Docker, or <code>ghcr.io</code> blocked? Run Open WebUI from pip instead</summary>

Open WebUI can run without Docker. Install it into its own virtual environment (Python
3.11+) and point it at the host API:

```bash
python -m venv openwebui-venv && source openwebui-venv/bin/activate
pip install open-webui

OPENAI_API_BASE_URL=http://localhost:8000/v1 \
OPENAI_API_KEY=sk-mygov-local \
open-webui serve --host 0.0.0.0 --port 3000
```

Then open http://localhost:3000. Use a different `--port` if 3000 is taken.
</details>

---

## Results

Retrieval quality on the labeled eval set (`data/eval_questions.json`: 24 on-topic
questions, 22 with a known target service, plus 6 off-topic). Off-topic recall is the share
of off-topic questions the gate refuses before any LLM call. Reproduce with
`python src/eval_run.py`.

| Method            |    hit@1     |    hit@5     |  MRR  | off-topic recall |
| ----------------- | :----------: | :----------: | :---: | :--------------: |
| hybrid            | 21/22 (0.95) | 22/22 (1.00) | 0.977 |    6/6 (1.00)    |
| hybrid + rerank   | 20/22 (0.91) | 22/22 (1.00) | 0.936 |    6/6 (1.00)    |

<sub>**hit@k / MRR** measure whether the correct service appears in the top-k retrieved
results. **Off-topic recall** measures how reliably the classifier gate refuses questions
that aren't about government services.</sub>

**Reading the numbers.** On this corpus both methods already place the right service in the
top 5 for every question, and plain hybrid retrieval is strong enough that the cross-encoder
doesn't improve top-1 here — on a set this clean, reranking mostly reshuffles items that are
already correct. The reranker earns its keep on larger, noisier candidate pools where the
first-stage ordering is less reliable; it is kept in the default path for that robustness,
and the grounding floor reads its score. The classifier gate refuses all six adversarial
questions (weather, small talk, arithmetic, insults) before any retrieval; each refusal is
then phrased by a single guarded fallback call (see *Refusals* below).

---

## Configuration

Set in `.env` (copied from `.env.example`). Only `OPENROUTER_API_KEY` is required.

| Variable             | Required | Default                                          | Purpose                                                                                                     |
| -------------------- | :------: | ------------------------------------------------ | ----------------------------------------------------------------------------------------------------------- |
| `OPENROUTER_API_KEY` |   yes    | —                                                | Powers the answer-generation and follow-up-rewrite LLM calls.                                                |
| `OPENROUTER_MODEL`   |    no    | `openrouter/free`                                | Which model answers and rewrites. The free auto-router is flaky (varies per call); **pin a real model** (e.g. `openai/gpt-4o-mini`, `google/gemini-2.0-flash-001`) for reliable multi-turn. |
| `DSN`                |    no    | `postgresql://postgres:mygov@localhost:5433/mygov` | Postgres connection string. Override to point at another DB or to keep the password out of source.        |
| `HF_TOKEN`           |    no    | —                                                | Raises Hugging Face download rate limits. Not needed once the models are cached.                            |
| `MYGOV_QUERY_LOG`    |    no    | — (off)                                          | Path to a JSONL file. When set, every gated decision (classifier prob, top rerank score, which gate fired) is appended — the raw data for `eval/calibrate.py`. No-op when unset. |

---

## Project structure

```
.
├── src/                     # served runtime
│   ├── api.py               # OpenAI-compatible FastAPI app (/v1/models, /v1/chat/completions, /health)
│   ├── rag.py               # embedder, DSN, LLM client, system prompt, vector retrieval
│   ├── hybrid.py            # vector + FTS + RRF fusion, typo-fix, connection pool
│   ├── rerank.py            # cross-encoder rerank + two-layer gate + follow-up rewrite (answer entrypoint)
│   └── eval_run.py          # hit@1 / hit@5 / MRR eval
├── eval/                    # gate-threshold calibration
│   ├── calibrate.py         # sweep CLF_OFFTOPIC / GATE_FLOOR on labeled data (no answer LLM)
│   └── labeled_seed.csv     # starter label set (grow it with MYGOV_QUERY_LOG traffic)
├── pipeline/                # offline corpus build (run once, in order)
│   ├── scrape_services.py   # my.gov.uz → data/services.jsonl
│   ├── chunk_with_synonyms.py  # chunk + inject synonyms → data/chunks.jsonl
│   ├── embed_store.py       # bge-m3 embed → Postgres/pgvector
│   └── build_vocab.py       # trigram vocab table for typo-fix
├── data/                    # shipped corpus + labeled eval set (see .gitignore for what's committed)
├── tests/                   # pytest critical-path suite
├── notebooks/               # exploration (not on the served path)
├── topic_clf.joblib         # trained on/off-topic classifier
├── docker-compose.yml       # Open WebUI frontend
└── requirements.txt         # pinned dependencies
```

---

## Tests

A small pytest suite proves the critical paths — typo-fix, the two-layer grounding gate,
retrieval shape, and classifier load. The LLM call is always mocked (no network), and tests
that need Postgres or the GPU models **skip cleanly** (with a reason) when those aren't
available, rather than faking a pass.

```bash
pytest -v
```

> **Sharing a GPU with the running API?** The API holds both models in VRAM, so a
> GPU-backed test run can OOM. Force the tests onto CPU: `CUDA_VISIBLE_DEVICES="" pytest -v`.

---

## Refusals & gate calibration

A query that can't be answered is refused, and the two failure modes are kept apart
(`src/rerank.py`):

- **`OUT_OF_SCOPE`** — the classifier judges it isn't a my.gov.uz topic (weather, sports, code).
- **`NO_SUPPORTING_CONTEXT`** — plausibly on-topic, but the reranker floor found nothing that
  grounds an answer.

Each refusal is written by `fallback_response()`: a single **guarded** LLM call that
acknowledges the question in its own language without answering it, and **degrades to a fixed
per-reason sentence** on any error, empty output, or over-long reply — so the refusal path
never errors, stalls on retries, or turns into a hallucination surface.

Both gates run on thresholds (`CLF_OFFTOPIC`, `GATE_FLOOR`) that should be **calibrated on
data, not guessed**. To do that:

1. Turn on logging in production: `MYGOV_QUERY_LOG=queries.jsonl` records each decision's
   classifier prob + top rerank score.
2. Label a set of questions (`question,label` CSV — see `eval/labeled_seed.csv` for the
   format and a starter set seeded from real services). Grow it with the logged traffic.
3. Sweep both thresholds against the labels — no answer LLM is called, so it's cheap:

   ```bash
   CUDA_VISIBLE_DEVICES="" ./mygov/bin/python eval/calibrate.py eval/labeled_seed.csv
   ```

   It reports, per threshold, the trade-off between false-accepts (a bad query admitted) and
   false-rejects (a good one refused), and suggests values that minimize total gate errors.

---

## Documentation

- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — service topology, the full request
  lifecycle, the Postgres data model, the offline corpus build, and the design decisions
  (and deliberate non-goals) behind the single-node deployment.
- **[AGENTS.md](AGENTS.md)** — operational guide for contributors and AI coding agents:
  setup, run, test, conventions, and the repo's real gotchas.

---

## License & data

The service content is scraped from the public [my.gov.uz](https://my.gov.uz) catalog and
belongs to its respective owners; it is included here to make the retrieval pipeline
reproducible. This project is for research and educational use.
