# Work Report — packaging my.gov.uz RAG as a portfolio repo

Date: 2026-08-24. Scope: audit the existing code, test/write the API, and add the
packaging files (env template, requirements, README, gitignore, tests) needed to publish
this as a reproducible GitHub portfolio repo. Not deployed. All work was done locally
against the running Postgres (`mygov-pg`, 926 chunks) and a CUDA GPU (RTX 3050 6 GB) with
both models already cached.

---

## 1. Audit findings (Task 1)

Read all 11 project `.py` files plus cross-checked the live DB schema. The pipeline works
end-to-end via the `rerank`/`hybrid` path. Findings, most to least important:

1. **`api.py` did not exist.** The brief described a "written, needs testing" FastAPI
   endpoint; there was no such file (and no FastAPI/uvicorn import anywhere). Resolved by
   writing it new — see §2.

2. **The live DB schema was not reproducible from any committed script.** Two objects the
   hybrid path depends on were built out-of-band:
   - the `keywords` column on `chunks` — read by `hybrid.py` (FTS expression, SELECT, and
     the rerank pair text), but **`embed_store.py`'s `CREATE TABLE` didn't define it and
     its `INSERT` didn't populate it**. Re-running the committed `embed_store.py` would
     have produced a DB that breaks `hybrid.py`.
   - the `vocab` trigram table + `pg_trgm` extension — the entire typo-fix
     (`hybrid.correct_query`) reads from `vocab`, but **no committed script created it**.

   Both resolved — see §2.

3. **`rag.py` crashed as a standalone script (tuple-arity bug).** `rag.retrieve` returned
   5-tuples but `rag.build_context` unpacks 6 (it expects `keywords`), so `rag.answer()`
   and `python rag.py` raised `ValueError`. Only the `rerank`/`hybrid` path (which returns
   6-tuples) worked. Fixed — see §2. Note this was a dead/legacy vector-only path; nothing
   else imported `rag.retrieve`, so the fix is self-contained.

4. **Minor:** importing `rag` (and therefore `hybrid`/`rerank`/`app`/`api`) requires
   `OPENROUTER_API_KEY` at import time and loads `bge-m3`; importing `rerank` additionally
   loads the 2.2 GB cross-encoder. Not a bug, but it shapes how the tests are structured
   (§4). Two redundant/superseded scripts also exist: `chunk_services.py` (v1 chunker,
   superseded by `chunk_with_synonyms.py`) and `scrape_mygov.py` (JSON-API scraper;
   `manifest.jsonl` is empty — the live data came from the HTML scraper
   `scrape_services.py`). Left as-is; they're harmless and show the project's evolution.

Per-file run status: see the table delivered in the Task 1 message. Summary — all scripts
run except the two issues above (`rag.py` standalone, and `embed_store.py` producing an
incomplete schema), both now fixed.

---

## 2. Changes made

### New files
| File | Purpose |
|------|---------|
| `notebooks/api.py` | **New.** OpenAI-compatible FastAPI wrapper (`POST /v1/chat/completions`, `GET /health`) over `rerank.answer`. Takes the last user message, runs the full gated pipeline unchanged, returns standard `choices` plus a non-standard `sources` array (empty on refusal). Lives in `notebooks/` so the bare `import rerank`→`rag`/`hybrid` chain resolves like `app.py`. Launch: `uvicorn api:app --app-dir notebooks --port 8000`. |
| `notebooks/build_vocab.py` | **New.** Builds the `vocab` trigram table the typo-fix needs (`word`, `ndoc`; `pg_trgm`; GIN trigram index) from the `chunks` corpus. Makes the typo-fix reproducible. See caveat below. |
| `.env.example` | Key template (`OPENROUTER_API_KEY=` empty; optional `HF_TOKEN`). No secrets. |
| `requirements.txt` | Pinned to the versions actually installed/imported across the repo. |
| `.gitignore` | Excludes `.env`, `__pycache__`, the `mygov/` venv, model caches, and scraping scratch/intermediate artifacts. Documents (in-file) the data files committed on purpose. |
| `README.md` | Project summary, prose architecture section, local setup (Docker → install → embed → Streamlit → API), empty Results table skeleton, `![screenshot]` markers. |
| `tests/conftest.py`, `tests/test_pipeline.py` | pytest suite (§4). |
| `WORK_REPORT.md` | This file. |

### Edited files
- **`notebooks/embed_store.py`** — added the `keywords TEXT` column to `CREATE TABLE` and
  populate it from `chunks.jsonl` (`r.get("keywords","")`) in the `INSERT`. Added a comment
  noting the `fts` column (built lazily by `hybrid.ensure_fts`) and `vocab` (built by
  `build_vocab.py`) are created downstream. **This makes the schema reproducible.**
- **`notebooks/rag.py`** — `retrieve` now selects `keywords` and returns 6-tuples matching
  `build_context`; fixed the `__main__` unpack to match. Verified: `rag.retrieve` +
  `build_context` now work (returned arity 6, context built).

### Environment
- Installed into the `mygov` venv (were missing): `fastapi`, `uvicorn[standard]`, `pytest`.
  All three are pinned in `requirements.txt`.

### Caveat on `build_vocab.py` (be aware)
The original vocab-building code was **not in the repo** — the live `vocab` (8,294 rows)
was built out-of-band. `build_vocab.py` is a faithful **reconstruction**, not a byte-exact
reproduction. Verified against the live table without overwriting it: core Uzbek words
match the live document-frequency exactly (`va`=925, `kadastr`=31), and `ikadastr` is
correctly absent so the typo-fix still fires. It differs from the current live table in two
documented ways: (a) it tokenizes on `\w+`, so the live table's decimal/hyphen tokens
(`4.78`, `arxitektura-rejalashtirish`) are split; (b) it *includes* the `keywords`
(synonym) field, so loanwords like `tonirovka` become correctable — a slight improvement.
**I did not overwrite the working live `vocab`.** Running `build_vocab.py` will rebuild a
functionally equivalent (~7.7k-word) table.

---

## 3. API test (Task 2) — PASSED, no fix needed

Started `uvicorn api:app --app-dir notebooks --port 8000` (both models loaded), curled both
queries against the live DB + real OpenRouter:

- **On-topic** `tonirovka ruxsatnomasi qanday olinadi?` → grounded answer citing service
  **#415** with URL, fee tiers, and 2-hour timing. Top rerank score **0.876** (≫ 0.05 floor).
- **Off-topic** `bugun ob-havo qanday?` → exact refusal `Menda bu haqda ishonchli ma'lumot
  yo'q.`, `sources: []`; the classifier gate fired before any retrieval or LLM call.

Both behaved correctly on the first run, so no pipeline code was changed for Task 2 (the
only new code is `api.py` itself).

---

## 4. Test results (Task 6) — 6 passed, 0 skipped, 0 failed (21.9 s)

`pytest -v`, run from repo root against the live DB + GPU. The LLM call is mocked in every
test (no network). DB- and model-dependent tests are written to **skip with a clear reason**
when Postgres or the models are unavailable — here everything was available, so all ran.

| Test | Covers | Needs |
|------|--------|-------|
| `test_typo_fix_corrects_out_of_vocab` | `ikadastr` → `kadastr` | DB (vocab) |
| `test_typo_fix_leaves_correct_word_unchanged` | `kadastr` left unchanged, no corrections | DB (vocab) |
| `test_offtopic_refuses_without_calling_llm` | off-topic (P<0.15) refuses; asserts the mocked LLM was **never** called | models |
| `test_ontopic_lowscore_passes_floor` | `pasport yo'qolsa` clears the 0.05 floor and reaches the (mocked) LLM; grounded rows returned | DB + models |
| `test_hybrid_retrieve_returns_6tuples` | `hybrid_retrieve` returns `(sid,title,url,text,keywords,score)`; types checked | DB + models |
| `test_classifier_loads_and_predicts` | `topic_clf.joblib` loads and yields a valid P(on-topic) | models |

Note: the brief expected `pasport yo'qolsa` to score ~0.073 at rerank; the measured value is
**0.121** (still above the 0.05 floor). The test asserts `> GATE_FLOOR` rather than a magic
number, so it's robust to that drift.

---

## 5. Eval snapshot (retrieval only, no LLM)

`python notebooks/eval_run.py` over the 22 labeled eval questions. Provided for reference —
the README Results table was intentionally left empty for you to fill from your own runs.

| Method          | hit@1        | hit@5        | MRR   |
|-----------------|:------------:|:------------:|:-----:|
| hybrid          | 21/22 (0.955)| 22/22 (1.000)| 0.977 |
| hybrid + rerank | 20/22 (0.909)| 22/22 (1.000)| 0.936 |

**Worth your attention:** on this small labeled set, the cross-encoder rerank *slightly
lowers* hit@1 (21→20) and MRR vs. hybrid alone. With only 22 labeled questions that's
within noise, but it's worth confirming on a larger eval set before claiming rerank
improves top-1 in the write-up. Off-topic recall (the fourth Results column) comes from the
classifier's held-out split — the last training run reported class-0 recall **1.00 on 21
held-out negatives**.

---

## 6. Verified vs. assumed — honest gaps

**Verified by running it here:**
- API on-topic (real OpenRouter call) and off-topic (refusal) — §3.
- All 6 tests pass against the live DB + GPU — §4.
- Retrieval eval numbers — §5.
- `rag.py` fix (`retrieve` + `build_context` return/consume 6-tuples).
- `build_vocab.py` logic validated against the live vocab (not overwritten) — §2 caveat.
- `embed_store.py` edit is a straightforward additive column; the corpus (`chunks.jsonl`)
  already carries `keywords`, so the populate path has real data.

**Assumed / not re-run (needs your call, or would mutate the working DB):**
- **`embed_store.py` and `build_vocab.py` were not executed against the live DB.** Doing so
  TRUNCATEs `chunks` / DROPs `vocab` — I didn't want to rebuild your working database while
  you're away. The edits are verified by inspection + the offline vocab validation, but a
  true from-scratch `embed_store.py → build_vocab.py` rebuild is unverified. Run them
  yourself when convenient to confirm the clean-build path.
- **`requirements.txt` pins were not tested in a fresh venv.** They're the versions actually
  installed in `mygov` and imported by the code, but I did not create a clean venv and
  `pip install -r` to confirm the set resolves together. `torch==2.13.0` in particular is a
  large platform-specific wheel (README notes the official-index fallback).
- **`python rag.py` end-to-end** (the standalone legacy script): retrieval + context build
  verified; the final LLM call within it is assumed working by analogy to the API path,
  which exercises the identical OpenRouter call.
- **Screenshots** in the README are placeholder `![screenshot]` markers — you'll add images.
- **README Results table** left empty by request.

**Nothing here required your GPU/DB/keys that I couldn't run** — they were all available, so
the live paths above are genuinely tested, not stubbed. The only deliberately-unrun items
are the destructive rebuild scripts and the fresh-venv install.

---

## 7. Recommendations (optional, not done)

- Run the from-scratch rebuild (`embed_store.py` → `build_vocab.py`) once to certify the
  clean-build path, then note it in the README.
- Consider whether the cross-encoder rerank earns its place given §5; expand the labeled
  eval set before deciding.
- Optionally delete the superseded `chunk_services.py` / `scrape_mygov.py`, or move them to
  an `archive/` folder, to keep the portfolio repo tight.

---

## 8. Multi-turn conversations (added 2026-08-25)

Added follow-up support as a single step *in front* of the grounding gate, leaving the
gate logic untouched (query-rewriting → unchanged classifier → retrieve → rerank → gate
→ answer).

### Changes
- **`notebooks/rerank.py`** — new `rewrite_query(history, question)`: turns a follow-up
  into a standalone query via a cheap `temperature=0` LLM call. Short-circuits to the raw
  question on the first turn (empty history) and falls back to it on any error.
  `answer` now takes `history=None`; it rewrites first, then **retrieves + gates on the
  standalone query** while the **final answer call also receives `history[-8:]`** for
  natural pronouns/tone. Added `HISTORY_TURNS = 8`.
- **`notebooks/api.py`** — `_last_user_question` → `_split_question_and_history`: the last
  non-empty user message is the question, everything before it is history (OpenWebUI/OpenAI
  clients already send the full `messages` array). Passed to `rerank.answer`.
- **`notebooks/app.py`** — Streamlit chat tab now keeps history in `st.session_state`,
  replays turns with `st.chat_message`, and has a "🗑 Yangi suhbat" clear button.
- **`notebooks/rag.py`** — `MODEL_LLM` now reads `OPENROUTER_MODEL` (default unchanged:
  `openrouter/free`). Documented in `.env.example` and README.

**No server-side session store (Redis) added** — sessions live in the UI/client layer.
Redis would only be needed for multi-instance server-side persistence across clients.

### Verified here (GPU free, live DB + real OpenRouter)
- **`tests/` suite: 6 passed, 0 failed.** The empty-history short-circuit means existing
  tests still see exactly one LLM call (`test_ontopic_lowscore_passes_floor` unchanged).
- **Reference resolution works — the failure is the free model, not the code.** Isolated
  `rewrite_query('narxi qancha?')` over 3 runs each:
  - *Clean* history (well-formed prior answer): **3/3** resolved to
    "Turar-joy kadastr pasporti narxi qancha?".
  - *Polluted* history (prior answer was the free model's garbage "User Safety: safe"):
    **2/3** — one run echoed the old question instead of resolving.
  - The live two-turn run failed because the free auto-router returned a garbage turn-1
    answer, which polluted history and then degraded the turn-2 rewrite: a cascade, one
    root cause.
- `temperature=0` is **not** deterministic under `openrouter/free` because it routes to a
  different underlying model per call.

### Known cost / caveat
- **+1 LLM call per turn** (the rewrite), on top of the answer call. Free by default.
- **Reliable multi-turn needs a pinned model** — set `OPENROUTER_MODEL` (e.g.
  `openai/gpt-4o-mini`). Left at `openrouter/free` per user's choice; the flakiness above
  is expected until a real model is pinned.

---

## Frontend switch: Streamlit → Open WebUI (2026-08-25)

Replaced the Streamlit demo with [Open WebUI](https://github.com/open-webui/open-webui)
as the chat frontend. Open WebUI is a standalone chat UI that speaks the OpenAI API, so it
talks to the existing `notebooks/api.py` over `/v1` — no bespoke UI code to maintain, and
it brings conversation history, model picker, and account management for free.

### Changes
- **`notebooks/api.py`** — added `GET /v1/models` (Open WebUI calls it to populate its
  model picker; returns the single `mygov-rag` model) and SSE **streaming** for
  `POST /v1/chat/completions` when `stream=true` (Open WebUI's default). The pipeline still
  produces the whole answer at once, so streaming emits it as one content delta then
  `[DONE]`; the non-stream JSON path is unchanged. `sources` is preserved on both paths.
- **`docker-compose.yml`** — **new.** Runs the Open WebUI container, mapped to
  `localhost:3000`, wired to the host API via
  `OPENAI_API_BASE_URL=http://host.docker.internal:8000/v1` (with a `host-gateway`
  `extra_hosts` entry so the container reaches the host on Linux). Data persisted in a named
  volume.
- **`notebooks/app.py`** — **removed** (the Streamlit chat + batch-eval tabs). The batch
  retrieval eval (hit@1/hit@5/MRR) already lives in `notebooks/eval_run.py`, so no eval
  capability was lost — only the Streamlit UI wrapper.
- **`requirements.txt`** — dropped `streamlit`. `pandas` kept (still used by the scraping /
  chunking / eval scripts).
- **`README.md`** — intro, session-handling paragraph, and run steps 5–6 rewritten: start
  the API, then `docker compose up -d` for Open WebUI at `http://localhost:3000`.

### Note
- The RAG API stays on the **host** (it needs the GPU, the cached models, and Postgres);
  only the thin UI is containerized. To run the API elsewhere, edit `OPENAI_API_BASE_URL`
  in `docker-compose.yml`.
