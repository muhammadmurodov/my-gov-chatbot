"""
Phase 2 - reranker. Cross-encoder re-scores hybrid candidates for sharper top-1.

Pipeline: hybrid (vector+FTS+RRF) -> 20 candidates -> cross-encoder -> best 5 -> LLM
  python rerank.py "notarius qabuliga qanday yozilaman?"
"""
import os
import re
import sys
import time
import joblib
from sentence_transformers import CrossEncoder
import rag
import hybrid

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
CLF_OFFTOPIC = 0.15        # classifier: below this P(on-topic), refuse
HISTORY_TURNS = 8          # how many prior messages to feed the rewriter / LLM


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
        return "Menda bu haqda ishonchli ma'lumot yo'q.", []

    rows = retrieve(standalone, k, pool)         # retrieve on the rewritten query
    if not rows or rows[0][-1] < GATE_FLOOR:     # grounding floor
        return "Menda bu haqda ishonchli ma'lumot yo'q.", rows

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