"""
Critical-path proof for the my.gov.uz RAG pipeline. Not exhaustive — just enough
to catch regressions in the parts that make answers trustworthy.

Design:
- The LLM call is always mocked, so tests never hit OpenRouter or the network.
- Tests that genuinely need Postgres or the GPU models SKIP (with a reason) when
  those aren't available, rather than faking a pass.

Run from the repo root:  pytest -v
"""
import os
import types
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
DSN = os.getenv("DSN", "postgresql://postgres:mygov@localhost:5433/mygov")


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def pipeline():
    """Import the model-backed modules once. Skip the whole model-backed suite if
    they can't load (no deps / no GPU / models not downloaded)."""
    try:
        import rag
        import hybrid
        import rerank
        import api
    except Exception as e:  # noqa: BLE001 - want the reason in the skip message
        pytest.skip(f"pipeline modules unavailable ({type(e).__name__}: {e})")
    return types.SimpleNamespace(rag=rag, hybrid=hybrid, rerank=rerank, api=api)


@pytest.fixture
def db_cursor():
    """A live cursor, or skip if Postgres isn't reachable."""
    import psycopg2
    try:
        conn = psycopg2.connect(DSN, connect_timeout=3)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Postgres not reachable on :5433 ({type(e).__name__})")
    cur = conn.cursor()
    yield cur
    cur.close()
    conn.close()


def _mock_llm(monkeypatch, pipeline, reply="MOCKED_ANSWER", recorder=None):
    """Replace the OpenRouter chat call. If recorder is a list, each call is logged;
    if reply is an Exception, calling raises it (used to prove 'no LLM call')."""
    def fake_create(*args, **kwargs):
        if recorder is not None:
            recorder.append(kwargs)
        if isinstance(reply, Exception):
            raise reply
        msg = types.SimpleNamespace(content=reply)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])
    monkeypatch.setattr(pipeline.rag._client.chat.completions, "create", fake_create)


# --------------------------------------------------------------------------- #
# 1. typo-fix
# --------------------------------------------------------------------------- #
def test_typo_fix_corrects_out_of_vocab(pipeline, db_cursor):
    fixed, changed = pipeline.hybrid.correct_query(db_cursor, "ikadastr pasporti")
    assert ("ikadastr", "kadastr") in changed
    assert "kadastr" in fixed.split()


def test_typo_fix_leaves_correct_word_unchanged(pipeline, db_cursor):
    fixed, changed = pipeline.hybrid.correct_query(db_cursor, "kadastr pasporti")
    assert changed == []
    assert fixed == "kadastr pasporti"


# --------------------------------------------------------------------------- #
# 2. two-layer grounding gate
# --------------------------------------------------------------------------- #
def test_offtopic_refuses_without_calling_llm(pipeline, monkeypatch):
    """Classifier gate (P<0.15) must refuse BEFORE any LLM call. No DB needed:
    the refusal happens before retrieval."""
    calls = []
    _mock_llm(monkeypatch, pipeline,
              reply=AssertionError("LLM must not be called for an off-topic query"),
              recorder=calls)
    ans, rows = pipeline.rerank.answer("bugun ob-havo qanday?")
    assert ans == "Menda bu haqda ishonchli ma'lumot yo'q."
    assert rows == []
    assert calls == []  # gate short-circuited before the LLM


def test_ontopic_lowscore_passes_floor(pipeline, db_cursor, monkeypatch):
    """On-topic query with a low-but-usable rerank score clears the 0.05 floor and
    reaches the (mocked) LLM."""
    calls = []
    _mock_llm(monkeypatch, pipeline, reply="MOCKED_ANSWER", recorder=calls)
    ans, rows = pipeline.rerank.answer("pasport yo'qolsa")
    assert ans == "MOCKED_ANSWER"                 # passed gate + floor -> LLM path
    assert rows                                   # grounded on real chunks
    assert rows[0][-1] > pipeline.rerank.GATE_FLOOR
    assert len(calls) == 1


# --------------------------------------------------------------------------- #
# 3. retrieval shape
# --------------------------------------------------------------------------- #
def test_hybrid_retrieve_returns_6tuples(pipeline, db_cursor):
    rows = pipeline.hybrid.hybrid_retrieve("kadastr pasporti qanday olinadi?", top_k=6)
    assert rows, "hybrid_retrieve returned no candidates"
    assert all(len(r) == 6 for r in rows), "expected (sid,title,url,text,keywords,score)"
    sid, title, url, text, keywords, score = rows[0]
    assert isinstance(sid, int)
    assert isinstance(url, str)
    assert isinstance(score, float)


# --------------------------------------------------------------------------- #
# 4. classifier loads + predicts
# --------------------------------------------------------------------------- #
def test_classifier_loads_and_predicts(pipeline):
    import joblib
    clf = joblib.load(ROOT / "topic_clf.joblib")
    emb = pipeline.rag._embedder.encode("pasport yo'qolsa", normalize_embeddings=True)
    proba = clf.predict_proba([emb])[0]
    assert len(proba) == 2                        # binary: off-topic / on-topic
    assert 0.0 <= float(proba[1]) <= 1.0


# --------------------------------------------------------------------------- #
# 5. RRF fusion (pure function — no DB/model)
# --------------------------------------------------------------------------- #
def test_rrf_fuses_and_ranks_by_summed_votes(pipeline):
    """An item ranked highly in BOTH lists must beat items ranked highly in only
    one, and every input id must survive the fusion."""
    vec = ["a", "b", "c"]
    fts = ["b", "a", "d"]
    fused = pipeline.hybrid.rrf(vec, fts)
    order = [cid for cid, _ in fused]
    scores = dict(fused)
    assert set(order) == {"a", "b", "c", "d"}      # union of both lists survives
    # a and b each sit at ranks 0 and 1 across the two lists → they tie, and both
    # outrank c and d, which appear in only one list.
    assert set(order[:2]) == {"a", "b"}
    assert scores["a"] == scores["b"]
    assert scores["b"] > scores["c"]               # in both lists > in one list
    assert scores["a"] > scores["d"]


# --------------------------------------------------------------------------- #
# 6. follow-up rewrite
# --------------------------------------------------------------------------- #
def test_rewrite_query_no_history_returns_raw(pipeline, monkeypatch):
    """First turn: nothing to resolve against, so the question is returned as-is
    WITHOUT an LLM call."""
    calls = []
    _mock_llm(monkeypatch, pipeline, reply="SHOULD_NOT_BE_USED", recorder=calls)
    out = pipeline.rerank.rewrite_query([], "kadastr pasporti narxi qancha?")
    assert out == "kadastr pasporti narxi qancha?"
    assert calls == []                             # no history → no rewrite call


def test_rewrite_query_uses_history(pipeline, monkeypatch):
    """A follow-up with history is rewritten by the (mocked) LLM into a standalone."""
    _mock_llm(monkeypatch, pipeline, reply="turar-joy kadastr pasporti narxi qancha?")
    history = [{"role": "user", "content": "kadastr pasporti qanday olinadi?"},
               {"role": "assistant", "content": "..."}]
    out = pipeline.rerank.rewrite_query(history, "narxi qancha?")
    assert out == "turar-joy kadastr pasporti narxi qancha?"


def test_rewrite_query_falls_back_on_error(pipeline, monkeypatch):
    """A rewriter failure must degrade to the raw question, not crash."""
    _mock_llm(monkeypatch, pipeline, reply=RuntimeError("rewriter down"))
    history = [{"role": "user", "content": "kadastr pasporti qanday olinadi?"}]
    out = pipeline.rerank.rewrite_query(history, "narxi qancha?")
    assert out == "narxi qancha?"                  # fell back to the raw question


def test_rewrite_skips_selfcontained_new_topic(pipeline, monkeypatch):
    """Regression: a self-contained question on a NEW topic must NOT be rewritten,
    so the previous topic can't leak in. Here the prior turn was about kadastr; the
    new question is a full tonirovka question and must pass through untouched, with
    no LLM rewrite call."""
    calls = []
    _mock_llm(monkeypatch, pipeline, reply="CONTAMINATED_BY_KADASTR", recorder=calls)
    history = [{"role": "user", "content": "kadastr haqida malumot ber"},
               {"role": "assistant", "content": "Kadastr xizmatlari ..."}]
    q = "tonirovka ruxsatnomasini olish necha pul"
    assert pipeline.rerank.rewrite_query(history, q) == q
    assert calls == []                             # self-contained -> no rewrite call


def test_needs_rewrite_classification(pipeline):
    """The follow-up gate: short/anaphoric questions rewrite; self-contained don't."""
    nr = pipeline.rerank._needs_rewrite
    assert nr("narxi qancha?")                     # 2 tokens -> elliptical
    assert nr("muddati?")                          # 1 token
    assert nr("uning narxi qancha")                # anaphora "uning"
    assert not nr("tonirovka ruxsatnomasini olish necha pul")  # self-contained
    assert not nr("kadastr haqida malumot ber")    # self-contained


# --------------------------------------------------------------------------- #
# 7. API layer — history filtering + endpoint contract (no DB/network)
# --------------------------------------------------------------------------- #
def test_split_keeps_only_user_assistant_history(pipeline):
    """The last user message is the question; history keeps only user/assistant
    turns so an Open WebUI system prompt can't stack onto SYSTEM_PROMPT."""
    api = pipeline.api
    msgs = [api.Message(role="system", content="Open WebUI injected prompt"),
            api.Message(role="user", content="birinchi savol"),
            api.Message(role="assistant", content="birinchi javob"),
            api.Message(role="user", content="oxirgi savol")]
    question, history = api._split_question_and_history(msgs)
    assert question == "oxirgi savol"
    assert history == [{"role": "user", "content": "birinchi savol"},
                       {"role": "assistant", "content": "birinchi javob"}]
    assert all(m["role"] in ("user", "assistant") for m in history)  # system dropped


def test_split_no_user_message_returns_none(pipeline):
    api = pipeline.api
    msgs = [api.Message(role="system", content="only a system message")]
    question, history = api._split_question_and_history(msgs)
    assert question is None
    assert history == []


def _client(pipeline):
    from fastapi.testclient import TestClient
    return TestClient(pipeline.api.app)


def test_api_chat_completion_shape(pipeline, monkeypatch):
    """Non-streaming happy path: rerank.answer is mocked, so no DB/LLM. The
    response must be OpenAI-shaped and carry the grounding sources."""
    rows = [(378, "ID karta", "https://my.gov.uz/uz/service/378", "matn", "kw", 0.9)]
    monkeypatch.setattr(pipeline.api.rerank, "answer",
                        lambda q, history=None: ("MOCKED_ANSWER", rows))
    r = _client(pipeline).post("/v1/chat/completions", json={
        "model": "mygov-rag",
        "messages": [{"role": "user", "content": "ID karta yo'qolsa?"}]})
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "MOCKED_ANSWER"
    assert body["sources"][0]["url"] == "https://my.gov.uz/uz/service/378"


def test_api_error_boundary_returns_502(pipeline, monkeypatch):
    """If the pipeline raises (DB down / LLM exhausted), the API returns a clean
    OpenAI-shaped 502 instead of leaking a 500 + stack trace."""
    def boom(q, history=None):
        raise RuntimeError("DB unreachable")
    monkeypatch.setattr(pipeline.api.rerank, "answer", boom)
    r = _client(pipeline).post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "savol"}]})
    assert r.status_code == 502
    assert r.json()["error"]["type"] == "upstream_error"


def test_api_no_user_message_returns_400(pipeline):
    r = _client(pipeline).post("/v1/chat/completions", json={
        "messages": [{"role": "system", "content": "no user turn"}]})
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "invalid_request_error"


def test_api_lists_the_model(pipeline):
    r = _client(pipeline).get("/v1/models")
    assert r.status_code == 200
    assert r.json()["data"][0]["id"] == pipeline.api.MODEL_ID
