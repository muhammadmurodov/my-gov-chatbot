"""
Phase 2 - reranker. Cross-encoder re-scores hybrid candidates for sharper top-1.

Pipeline: hybrid (vector+FTS+RRF) -> 20 candidates -> cross-encoder -> best 5 -> LLM
  python rerank.py "notarius qabuliga qanday yozilaman?"
"""
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

import joblib
from sentence_transformers import CrossEncoder
import rag
import hybrid

# Optional query log: set MYGOV_QUERY_LOG=/path/to/log.jsonl to record every gated
# decision (classifier prob + top rerank score + which gate fired). This is the raw
# material for calibrating CLF_OFFTOPIC / GATE_FLOOR on real traffic (see eval/calibrate.py).
# Off by default — when the env var is unset, _log_query is a no-op with zero overhead.
QUERY_LOG = os.getenv("MYGOV_QUERY_LOG")


def _log_query(question, standalone, clf_prob, top_score, decision):
    """Append one decision record as JSONL. Best-effort: logging must never break or
    slow the answer path, so any failure (bad path, disk full) is swallowed."""
    if not QUERY_LOG:
        return
    try:
        rec = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "question": question,
            "standalone": standalone,
            "clf_prob": round(float(clf_prob), 4),
            "top_score": None if top_score is None else round(float(top_score), 4),
            "decision": decision,   # OUT_OF_SCOPE | NO_SUPPORTING_CONTEXT | ANSWERED
        }
        with open(QUERY_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass

# cross-encoder: reads (query, chunk) TOGETHER and outputs one relevance score.
# ~2.2 GB download first run; runs on your GPU.
_reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)

# topic classifier: LogisticRegression over bge-m3 embeddings, P(on-topic).
# lives at project root; load via __file__ so cwd doesn't matter.
_CLF_PATH = os.path.join(os.path.dirname(__file__), "..", "topic_clf.joblib")
_clf = joblib.load(_CLF_PATH)


def rerank(question, rows, top_k=5):
    pairs = [(question, f"{r[1]}\n{r[4]}\n{r[3]}") for r in rows]  # title + keywords + text
    scores = _reranker.predict(pairs)
    ranked = sorted(zip(rows, scores), key=lambda x: x[1], reverse=True)
    return [(sid, title, url, text, kw, float(s))
            for (sid, title, url, text, kw, _), s in ranked[:top_k]]


def retrieve(question, top_k=5, pool=20):
    candidates = hybrid.hybrid_retrieve(question, top_k=pool)  # broad, cheap
    return rerank(question, candidates, top_k)                 # precise, narrow


GATE_FLOOR = 0.05          # grounding floor: below this, retrieval found nothing usable
CLF_OFFTOPIC = 0.20        # classifier: below this P(on-topic), refuse. Raised 0.15->0.20
                           # from eval/calibrate.py — at 0.15 two gov-but-not-in-corpus
                           # questions leaked to the LLM; in_domain sits at clf>=0.57, so
                           # 0.20 closes the leak with no false-rejects (seed set, directional).
HISTORY_TURNS = 8          # how many prior messages to feed the rewriter / LLM

# Two fundamentally different refusals — kept apart instead of collapsing into one
# fixed sentence. OUT_OF_SCOPE: the question isn't about my.gov.uz at all (weather,
# Python, football). NO_SUPPORTING_CONTEXT: it plausibly *is* on-topic, but retrieval
# found nothing that grounds an answer. Each gets its own reply and its own tone.
OUT_OF_SCOPE = "out_of_scope"
NO_SUPPORTING_CONTEXT = "no_supporting_context"

# Deterministic hard fallbacks. fallback_response() asks the LLM for a natural,
# question-aware reply, but on ANY failure it degrades to these — so the refusal
# path itself can never error, hang, or hallucinate. NO_SUPPORTING_CONTEXT keeps the
# original fixed sentence.
_HARD_FALLBACK = {
    OUT_OF_SCOPE: (
        "Bu savol my.gov.uz davlat xizmatlari doirasidan tashqarida. Men my.gov.uz "
        "saytidagi davlat xizmatlari va ularga oid ma'lumotlar bo'yicha yordam bera olaman."
    ),
    NO_SUPPORTING_CONTEXT: "Menda bu haqda ishonchli ma'lumot yo'q.",
}

# The situation description handed to the fallback LLM per reason.
_FALLBACK_SITUATION = {
    OUT_OF_SCOPE: "Foydalanuvchi savoli my.gov.uz davlat xizmatlari mavzusidan tashqarida.",
    NO_SUPPORTING_CONTEXT: ("Savol my.gov.uz mavzusiga aloqador bo'lishi mumkin, lekin unga "
                            "javob beradigan ma'lumot mavjud hujjatlarda topilmadi."),
}

# A fallback reply must acknowledge the question without answering it. Cap the length:
# a chatty model that starts smuggling a real answer runs long, so an over-long reply
# is treated as a leak and swapped for the hard fallback.
_FALLBACK_MAX_CHARS = 400


def fallback_response(question, reason):
    """A natural, question-aware refusal that acknowledges what was asked WITHOUT
    answering it, then points the user back to my.gov.uz services.

    Guarded on purpose: a single deterministic-ish LLM call (no retry loop — the
    refusal path must stay fast on off-topic spam), and on any exception, empty
    output, or a suspiciously long reply it degrades to the fixed per-reason string
    in _HARD_FALLBACK. So this path never becomes a source of errors, latency
    blow-ups, or hallucinated answers.
    """
    hard = _HARD_FALLBACK.get(reason, _HARD_FALLBACK[NO_SUPPORTING_CONTEXT])
    situation = _FALLBACK_SITUATION.get(reason, _FALLBACK_SITUATION[NO_SUPPORTING_CONTEXT])
    prompt = (
        "Sen my.gov.uz davlat xizmatlari bo'yicha AI yordamchisan.\n\n"
        f"Foydalanuvchi savoli:\n{question}\n\n"
        f"Vaziyat:\n{situation}\n\n"
        "Vazifa:\n"
        "- Savolning O'ZIGA JAVOB BERMA. Hech qanday fakt, ma'lumot yoki taxmin berma.\n"
        "- Savol mavzusiga mos ohangda, tabiiy va qisqa tarzda tushuntir nega bunga "
        "javob bera olmasligingni.\n"
        "- Foydalanuvchi my.gov.uz davlat xizmatlari bo'yicha savol berishi mumkinligini ayt.\n"
        "- 1-3 ta gapdan oshirma. Savol qaysi tilda bo'lsa (o'zbek/rus), o'sha tilda javob ber.\n"
        "- Faqat javob matnini qaytar, boshqa hech narsa yozma."
    )
    try:
        resp = rag._client.chat.completions.create(
            model=rag.MODEL_LLM,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            timeout=15,                  # fail fast to the hard fallback, don't stall a refusal
        )
        text = (resp.choices[0].message.content or "").strip()
    except Exception:
        return hard                      # network/LLM down -> deterministic reply
    if not text or len(text) > _FALLBACK_MAX_CHARS:
        return hard                      # empty, or long enough to look like a real answer
    return text


# Anaphora / ellipsis cues that mark a question as leaning on prior turns — a
# genuine follow-up ("uning narxi qancha?", "а сколько стоит?"). Rewriting only
# fires for these (or very short questions); a self-contained new-topic question
# like "tonirovka ruxsatnomasi necha pul" is left alone, so the previous topic
# can't be dragged into it by the rewriter.
_FOLLOWUP_CUES = {
    # uz
    "uning", "buning", "shuning", "uni", "buni", "shuni", "unga", "bunga",
    "undan", "bundan", "u", "bu", "shu", "o'sha", "osha", "yuqoridagi",
    "yana", "ham", "-chi", "chi", "va",
    # ru
    "его", "ее", "её", "их", "это", "этот", "эта", "том", "нем", "нём",
    "туда", "тоже", "также", "а",
}
# At or below this many word tokens a question is treated as elliptical (e.g.
# "narxi qancha?", "muddati?") and rewritten against history.
_REWRITE_MAX_STANDALONE_TOKENS = 2


def _needs_rewrite(question):
    """True if the question looks context-dependent (a genuine follow-up) and so
    should be rewritten against history; False if it is already self-contained."""
    toks = re.findall(r"\w+", question.lower())
    if len(toks) <= _REWRITE_MAX_STANDALONE_TOKENS:
        return True                                  # very short -> likely elliptical
    return any(t in _FOLLOWUP_CUES for t in toks)    # explicit anaphora/continuation


def rewrite_query(history, question):
    """Turn a follow-up into a standalone query using conversation history.

    history = list of {"role","content"} for PRIOR turns (not incl. current).
    Returns the raw question unchanged on the first turn, when the question is
    already self-contained (see _needs_rewrite — avoids dragging the prior topic
    into a new-topic question), or if the rewrite call fails — so a rewriter
    failure degrades to single-turn behavior instead of crashing.
    """
    if not history or not _needs_rewrite(question):
        return question

    convo = "\n".join(f"{m['role']}: {m['content']}" for m in history[-HISTORY_TURNS:])
    prompt = (
        "Quyidagi suhbat asosida foydalanuvchining oxirgi savolini "
        "mustaqil, to'liq savolga aylantir. Faqat qayta yozilgan savolni qaytar, "
        "boshqa hech narsa yozma. Agar savol allaqachon mustaqil bo'lsa, "
        "uni o'zgartirmasdan qaytar.\n\n"
        f"SUHBAT:\n{convo}\n\nOXIRGI SAVOL: {question}\n\nQAYTA YOZILGAN SAVOL:"
    )
    try:
        resp = rag._client.chat.completions.create(
            model=rag.MODEL_LLM,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,             # deterministic — rewriting is mechanical
        )
        rewritten = (resp.choices[0].message.content or "").strip()
        return rewritten or question     # empty completion -> fall back to raw
    except Exception:
        return question                  # rewriter down -> single-turn behavior


def answer(question, history=None, k=5, pool=20):
    history = history or []
    standalone = rewrite_query(history, question)   # resolve references first

    # NOTE: micro-opt for later — emb is embedded here and again inside
    # retrieve->hybrid_retrieve (one extra embed/query). Fine until latency matters.
    emb = rag._embedder.encode(standalone, normalize_embeddings=True)
    p = _clf.predict_proba([emb])[0][1]

    if p < CLF_OFFTOPIC:                         # classifier owns off-topic
        _log_query(question, standalone, p, None, OUT_OF_SCOPE)
        return fallback_response(standalone, OUT_OF_SCOPE), []

    rows = retrieve(standalone, k, pool)         # retrieve on the rewritten query
    top_score = rows[0][-1] if rows else None
    if not rows or top_score < GATE_FLOOR:       # grounding floor
        _log_query(question, standalone, p, top_score, NO_SUPPORTING_CONTEXT)
        # Nothing here grounded an answer, so expose no sources — 5 citations under a
        # "not found" reply would falsely imply the answer rests on those services.
        return fallback_response(standalone, NO_SUPPORTING_CONTEXT), []

    _log_query(question, standalone, p, top_score, "ANSWERED")

    # Retrieval uses the standalone query (right chunks); the answer call ALSO gets
    # history (natural pronouns/tone). Two different context needs, handled apart.
    user_msg = f"KONTEKST:\n{rag.build_context(rows)}\n\nSAVOL: {standalone}"
    messages = [{"role": "system", "content": rag.SYSTEM_PROMPT}]
    messages += history[-HISTORY_TURNS:]
    messages.append({"role": "user", "content": user_msg})

    last = None
    for attempt in range(4):
        try:
            resp = rag._client.chat.completions.create(
                model=rag.MODEL_LLM,
                messages=messages,
                temperature=0.2,
            )
            return resp.choices[0].message.content, rows
        except Exception as e:
            last = e
            time.sleep(2 * (attempt + 1))
    raise last


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "notarius qabuliga qanday yozilaman?"
    print(f"\nQUERY: {q}\n" + "=" * 60)
    for sid, title, url, _, _, score in retrieve(q):
        print(f"  [{score:+.3f}] #{sid}  {title}")