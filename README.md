# my.gov.uz RAG — Uzbek government-services Q&A

A retrieval-augmented question-answering system over the public service catalog of
[my.gov.uz](https://my.gov.uz). It answers questions in Uzbek and Russian about how to
obtain government services — documents, fees, deadlines, steps — grounded strictly in
the scraped service pages, and **refuses** (rather than guessing) when a question is
off-topic or unsupported by the corpus. Everything is hand-rolled Python — no LangChain,
no vector-DB SaaS — running locally against Postgres + pgvector, with `bge-m3` for
embeddings and `bge-reranker-v2-m3` for reranking. It ships as an OpenAI-compatible
HTTP API, fronted by an [Open WebUI](https://github.com/open-webui/open-webui) chat
interface (run via Docker) that talks to the API over its `/v1` endpoints.

![screenshot] <!-- Open WebUI chat answering a service question with a citation -->

## Architecture

The corpus is built by scraping the public, server-rendered service pages
(`pipeline/scrape_services.py`), one JSON record per service. Those records are split
into ~500-token retrieval chunks that break on sentence boundaries with a small overlap
(`pipeline/chunk_with_synonyms.py`); the same step injects a hand-curated synonym map so
loanwords people actually search for — *tonirovka*, *propiska* — become findable even
when the official page uses different wording. Each chunk is embedded with `bge-m3` (1024
dimensions, cosine) and stored in Postgres/pgvector alongside its title, URL, and
keywords (`pipeline/embed_store.py`).

Retrieval is hybrid. A query is answered by two independent searches over the chunk
table: a vector nearest-neighbour search that matches on *meaning*, and a Postgres
full-text search that matches on *exact words*. Their ranked lists are fused with
Reciprocal Rank Fusion, so a chunk that scores well on either signal surfaces
(`src/hybrid.py`). Before the full-text search runs, every out-of-vocabulary query
token is spell-corrected against a trigram index of the real corpus vocabulary
(`pipeline/build_vocab.py`), so a typo like *ikadastr* still finds *kadastr*. The fused
candidate pool is then re-scored by the `bge-reranker-v2-m3` cross-encoder, which reads
each (query, chunk) pair together and produces a sharper ordering than either first-stage
signal alone (`src/rerank.py`).

What makes the answers trustworthy is a two-layer grounding gate in front of the LLM.
First, a logistic-regression classifier over the query embedding estimates the
probability the question is even about government services; below a threshold the system
refuses immediately, without retrieving or calling the LLM — this is what catches
"what's the weather today?". Second, if retrieval does run but the best reranked chunk
scores below a grounding floor, the system refuses rather than answer from weak context.
Only when both gates pass is the top context handed to the LLM (via OpenRouter) with a
strict system prompt: answer *only* from the provided context, always cite the service
name and URL, and emit a fixed refusal sentence when the context doesn't contain the
answer. The classifier is trained separately and saved to `topic_clf.joblib`.

Follow-up questions are handled by a single step added *in front* of that gate, leaving
the grounding logic untouched. Before anything else, a cheap LLM call rewrites the latest
message into a standalone query using the recent conversation, so *"narxi qancha?"* after
a question about kadastr passports becomes *"turar-joy kadastr pasporti narxi qancha?"*
(`rewrite_query` in `src/rerank.py`). The two context needs are then met
separately: **retrieval and both gates run on the rewritten standalone query** (so they
find the right chunks), while the **final answer call also receives the prior turns** (so
pronouns and tone stay natural). Sessions live in the UI layer — Open WebUI keeps each
conversation and sends the full `messages` array on every request, and the API simply
uses that array — so no server-side session store (e.g. Redis) is needed; that would only
be added for multi-instance, server-side persistence across clients. The rewrite is a mechanical task run at `temperature=0`, and it degrades to the
raw question on any error, so a rewriter hiccup falls back to single-turn behaviour rather
than crashing. Because the rewrite is an extra LLM call per turn, reliable multi-turn
depends on a real model — see `OPENROUTER_MODEL` below; the free auto-router produces
flaky rewrites and low-quality turns that then pollute the history.

![screenshot] <!-- Architecture / pipeline diagram -->

## How it works locally

Everything runs on one machine. A CUDA GPU is recommended (the reranker is a
cross-encoder); CPU works but is slower. The two models download (~2.2 GB each) on first
run and are cached afterward.

**1 — Start Postgres + pgvector (Docker).** The code expects it on port 5433:

```bash
docker run -d --name mygov-pg \
  -e POSTGRES_PASSWORD=mygov -e POSTGRES_DB=mygov \
  -p 5433:5432 pgvector/pgvector:pg16
```

This matches the DSN in the code: `postgresql://postgres:mygov@localhost:5433/mygov`.

**2 — Install dependencies** into a virtual environment:

```bash
python -m venv mygov && source mygov/bin/activate
pip install -r requirements.txt
```

**3 — Add your API key.** Copy the template and fill in an OpenRouter key (used only for
the final answer-generation step):

```bash
cp .env.example .env
# edit .env -> OPENROUTER_API_KEY=...
# optional: OPENROUTER_MODEL=openai/gpt-4o-mini   # pin a real model (see below)
```

The answer + follow-up-rewrite model defaults to the free `openrouter/free` auto-router,
which is convenient but flaky — it routes to a different model per call, so `temperature=0`
isn't actually deterministic and weak turns can pollute the conversation. For reliable
multi-turn, set `OPENROUTER_MODEL` to a pinned model (e.g. `openai/gpt-4o-mini` or
`google/gemini-2.0-flash-001`).

**4 — Build the index.** The repo already ships the corpus (`data/chunks.jsonl`), so you
can skip scraping. Embed the chunks into Postgres, then build the typo-fix vocabulary.
**Run every command from the repo root** — data paths are resolved relative to the
working directory, and the `src/` app modules import each other by bare name:

```bash
python pipeline/embed_store.py     # embed 926 chunks -> pgvector (TRUNCATEs + reloads)
python pipeline/build_vocab.py     # build the vocab trigram table for typo-fix
```

<details>
<summary>Optional: rebuild the corpus from scratch instead of using the shipped data</summary>

```bash
python pipeline/scrape_services.py                       # -> data/services.jsonl
python pipeline/chunk_with_synonyms.py \
    data/services.jsonl data/synonyms.jsonl data/chunks.jsonl   # -> data/chunks.jsonl
```
</details>

**5 — Start the API** (OpenAI-compatible; this is the backend the chat UI talks to).
Keep it running in its own terminal. Bind to `0.0.0.0` (not the default `127.0.0.1`) so
the Open WebUI container can reach it through `host.docker.internal`:

```bash
uvicorn api:app --app-dir src --host 0.0.0.0 --port 8000
```

It exposes `GET /v1/models` and `POST /v1/chat/completions`. You can hit it directly:

```bash
curl -s localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"mygov-rag","messages":[{"role":"user","content":"tonirovka ruxsatnomasi qanday olinadi?"}]}'
```

On-topic queries return a grounded answer plus the services that grounded it; off-topic
queries (`"bugun ob-havo qanday?"`) return the fixed refusal, `Menda bu haqda ishonchli
ma'lumot yo'q.`

**6 — Start the Open WebUI chat frontend** (Docker). It runs in a container and forwards
to the host API from step 5:

```bash
docker compose up -d      # then open http://localhost:3000
```

On first visit, Open WebUI asks you to create a local admin account (stored in a Docker
volume, not sent anywhere). The `mygov-rag` model then appears in the model picker
automatically — pick it and chat. The connection is wired in `docker-compose.yml`
(`OPENAI_API_BASE_URL=http://host.docker.internal:8000/v1`); if you run the API on a
different host or port, edit that value.

![screenshot] <!-- Open WebUI chat with the mygov-rag model selected -->

> Batch retrieval eval (hit@1 / hit@5 / MRR) lives in `src/eval_run.py`, run
> directly from the repo root — see [Results](#results).

**Tests.** A small pytest suite proves the critical paths (typo-fix, the grounding gate,
retrieval shape, classifier load). DB/GPU-dependent tests skip cleanly when those aren't
present:

```bash
pytest -v
```

## Results

Retrieval quality on the labeled eval set (`data/eval_questions.json`), and off-topic
recall from the classifier's held-out test split. _Numbers to be filled from eval runs
(`python src/eval_run.py`)._

| Method          | hit@1 | hit@5 | MRR  | off-topic recall |
|-----------------|:-----:|:-----:|:----:|:----------------:|
| hybrid          |       |       |      |        —         |
| hybrid + rerank |       |       |      |        —         |

<sub>hit@k / MRR measure whether the correct service appears in the top-k retrieved
results. Off-topic recall measures how reliably the classifier gate refuses questions
that aren't about government services.</sub>

![screenshot] <!-- Eval results table / chart -->
