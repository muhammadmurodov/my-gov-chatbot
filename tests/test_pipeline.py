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
    except Exception as e:  # noqa: BLE001 - want the reason in the skip message
        pytest.skip(f"pipeline modules unavailable ({type(e).__name__}: {e})")
    return types.SimpleNamespace(rag=rag, hybrid=hybrid, rerank=rerank)


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
