# Architecture

This document describes the runtime and the offline corpus build for the my.gov.uz RAG
system, and the request lifecycle of a single question. It reflects the **implemented**
system — a single-node deployment whose only external dependency is a hosted LLM
(OpenRouter). Where this document and the code disagree, the code wins — open an issue.

---

## 1. Service topology

```mermaid
flowchart TB
    user([User])

    subgraph node["Single host"]
        direction TB
        webui["Open WebUI<br/><i>chat frontend</i><br/>container · :3001"]
        api["FastAPI API<br/><i>OpenAI-compatible /v1</i><br/>src/api.py · :8000"]

        subgraph models["In-process models (sentence-transformers)"]
            direction LR
            emb["bge-m3<br/>embeddings · 1024-d"]
            rr["bge-reranker-v2-m3<br/>cross-encoder"]
        end

        clf["topic classifier<br/>topic_clf.joblib"]
        pg[("Postgres + pgvector + FTS<br/>container · :5433<br/>chunks · vocab")]
    end

    llm["OpenRouter<br/><i>hosted LLM</i><br/>openrouter.ai/api/v1"]

    user -->|"HTTP"| webui
    webui -->|"POST /v1/chat/completions"| api
    api --> emb
    api --> rr
    api --> clf
    api -->|"vector + full-text SQL"| pg
    api -->|"chat completions (HTTPS)"| llm
```

**Why it is this small.** The LLM is hosted (OpenRouter), so there is no local
inference server to run. The embedding and reranker models are loaded once into the API
process via `sentence-transformers` — a separate embedding microservice would add a network
hop and a second copy of the weights for no benefit at this scale. Conversation state lives
in the UI: Open WebUI stores each chat and replays the full `messages` array every request,
so there is no server-side session store (no Redis). Everything scales vertically on one box
until traffic says otherwise.

| Component | Role | Port | Where |
| --- | --- | :---: | --- |
| Open WebUI | Chat frontend; holds sessions; speaks OpenAI `/v1` | 3001 | Docker (`mygov-webui`) |
| FastAPI API | `GET /v1/models`, `POST /v1/chat/completions`, `GET /health` | 8000 | `src/api.py` |
| Postgres + pgvector | Chunk store (vector + FTS) and typo-fix vocabulary | 5433 | Docker (`mygov-pg`) |
| bge-m3 | Query/passage embeddings (1024-d, cosine) | — | in-process |
| bge-reranker-v2-m3 | Cross-encoder reranking of candidates | — | in-process |
| topic classifier | Logistic regression over the query embedding | — | `topic_clf.joblib` |
| OpenRouter | Answer generation + follow-up rewrite | — | hosted |

---

## 2. Request lifecycle

The entrypoint is `rerank.answer(question, history)`, called by `POST /v1/chat/completions`
(`src/api.py`). The two grounding gates run **before** the LLM is ever called.

```mermaid
flowchart TD
    A["POST /v1/chat/completions<br/>(last user msg = question, prior turns = history)"] --> B

    B["1 · rewrite_query<br/>follow-up → standalone<br/>(only if anaphoric; anchored to<br/>services cited last turn)"] --> C
    C["2 · embed (bge-m3) + topic classifier"] --> D{"3 · off-topic gate<br/>P(on-topic) ≥ 0.20?"}
    D -- no --> REF["fallback_response(OUT_OF_SCOPE)<br/>reason-aware refusal, no sources"]
    D -- yes --> E["4 · hybrid retrieve<br/>vector NN + Postgres FTS → RRF<br/>(OOV tokens typo-fixed first)"]
    E --> F["5 · rerank<br/>bge-reranker-v2-m3 cross-encoder → top 5"]
    F --> G{"6 · grounding floor<br/>top score ≥ 0.05?"}
    G -- no --> REF2["fallback_response(NO_SUPPORTING_CONTEXT)<br/>reason-aware refusal, no sources"]
    G -- yes --> H["7 · assemble context<br/>top services' text + strict system prompt<br/>(+ history only if follow-up)"]
    H --> I["8 · LLM answer (OpenRouter)<br/>retry ×4; retries on empty content"]
    I --> J["Grounded answer + service citations<br/>(non-standard sources field)"]
```

| # | Step | Code | Notes |
| :-: | --- | --- | --- |
| 1 | Follow-up rewrite | `rewrite_query` | Fires only for anaphoric/very short questions; anchored to the services cited in the previous answer so *"bu xizmat"* resolves to the primary one. Runs at `temperature=0`; degrades to the raw question on any error. |
| 2 | Embed + classify | `rag._embedder`, `topic_clf` | One `bge-m3` embedding feeds both the classifier and retrieval. |
| 3 | Off-topic gate | `CLF_OFFTOPIC = 0.20` | Refuses *before* retrieval/LLM — catches "what's the weather?". |
| 4 | Hybrid retrieve | `src/hybrid.py` | Vector nearest-neighbour (meaning) + Postgres full-text (exact words), fused with Reciprocal Rank Fusion. OOV query tokens are spell-corrected against the `vocab` trigram table first. |
| 5 | Rerank | `rerank()` | Cross-encoder reads each (query, chunk) pair together → sharper top-5. |
| 6 | Grounding floor | `GATE_FLOOR = 0.05` | If the best reranked chunk scores below the floor, refuse instead of answering from weak context. |
| 7 | Context assembly | `rag.build_context` | Only the top services' plain text + a strict system prompt. History is included **only for a genuine follow-up**, to avoid a topic switch dragging the previous topic into the answer. |
| 8 | LLM answer | `rag._client` (OpenRouter) | Answer strictly from context; always cite service name + URL. Retries up to 4× and treats empty content as retriable. |

Both refusal paths return **no** `sources`, so a "not found" reply never shows misleading
citations. The two reasons are kept apart on purpose:

- **`OUT_OF_SCOPE`** — the classifier judges the question isn't a my.gov.uz topic.
- **`NO_SUPPORTING_CONTEXT`** — plausibly on-topic, but retrieval found nothing that grounds
  an answer.

Each refusal is phrased by a single **guarded** LLM call (`fallback_response`) that
acknowledges the question without answering it and **degrades to a fixed per-reason
sentence** on any error, empty output, or over-long reply — so the refusal path can never
error, stall, or hallucinate.

---

## 3. Data model (Postgres)

Built by the offline pipeline; read by `src/hybrid.py` and `src/rag.py`.

```
chunks
  chunk_id    INTEGER PRIMARY KEY   -- one row per ~500-token chunk
  service_id  INTEGER NOT NULL      -- my.gov.uz service the chunk belongs to
  title       TEXT                  -- service title (prepended to embed_text)
  url         TEXT                  -- canonical my.gov.uz/uz/service/<id> link
  lang        TEXT                  -- uz / ru
  keywords    TEXT                  -- injected synonyms + salient terms
  text        TEXT                  -- chunk body
  embedding   vector(1024)          -- bge-m3, cosine        → ANN index chunks_emb_idx
  fts         tsvector GENERATED    -- title+keywords+text   → GIN index (full-text search)

vocab                               -- powers query typo-fix (pg_trgm)
  word        TEXT PRIMARY KEY      -- every real corpus token
  ndoc        INTEGER               -- document frequency    → GIN trigram index on word
```

Retrieval issues two queries over `chunks` — `ORDER BY embedding <=> $qv` (vector) and
`ORDER BY ts_rank(fts, plainto_tsquery(...))` (full-text) — and fuses their ranked lists
with RRF. `vocab` is consulted only to repair out-of-vocabulary query tokens before the
full-text search runs.

---

## 4. Offline corpus build

Run once, in order, from the repo root. The repo ships `data/chunks.jsonl`, so scraping is
optional.

```mermaid
flowchart LR
    S["scrape_services.py<br/>my.gov.uz pages"] --> J["services.jsonl"]
    J --> C["chunk_with_synonyms.py<br/>~500-token chunks + synonym map"]
    C --> K["chunks.jsonl"]
    K --> E["embed_store.py<br/>bge-m3 → pgvector"] --> PG[("chunks")]
    K --> V["build_vocab.py<br/>trigram vocabulary"] --> PG2[("vocab")]
```

- **`scrape_services.py`** — server-rendered service pages → one JSON record per service.
- **`chunk_with_synonyms.py`** — split into ~500-token chunks on sentence boundaries with a
  small overlap; inject a hand-curated synonym map so loanwords people actually type
  (*tonirovka*, *propiska*) become findable.
- **`embed_store.py`** — embed each chunk with `bge-m3` and `TRUNCATE`-then-reload the
  `chunks` table.
- **`build_vocab.py`** — rebuild the `vocab` trigram table from the corpus (idempotent).

---

## 5. Design decisions & non-goals

- **Hosted LLM, not local inference.** OpenRouter removes a GPU-serving component from the
  runtime. The two retrieval models are small enough to co-locate in the API process.
- **UI-held sessions.** Open WebUI replays full history each turn, so the server stays
  stateless — no session store, no Redis.
- **Gates before the model.** Trust comes from refusing early (classifier) and refusing on
  weak evidence (grounding floor), not from asking the LLM to police itself.
- **Thresholds are calibrated, not guessed.** `CLF_OFFTOPIC` and `GATE_FLOOR` are tuned from
  logged decisions (`MYGOV_QUERY_LOG` → `eval/calibrate.py`).
- **Not in scope (single-node design):** an API gateway / rate-limiter tier, a separate
  embedding/rerank microservice, a local inference server, and a distributed session store.
  These belong to a horizontally-scaled deployment; this repo targets one host.
