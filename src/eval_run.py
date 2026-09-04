"""
Phase 2 eval - hit@1, hit@5, MRR for hybrid vs hybrid+rerank. No LLM calls.
  python eval_run.py
"""
import json
import hybrid
import rerank


def rank_of(expected, rows):
    ids = [r[0] for r in rows]
    return ids.index(expected) + 1 if expected in ids else 0  # 0 = not found


def score(method_fn, questions, k=10):
    h1 = h5 = n = 0
    mrr = 0.0
    for it in questions:
        exp = it.get("expected_service_id")
        if exp is None:
            continue
        n += 1
        r = rank_of(exp, method_fn(it["query"], k))
        if r == 1:
            h1 += 1
        if r and r <= 5:
            h5 += 1
        if r:
            mrr += 1 / r          # MRR: 1/rank of the correct service
    return h1, h5, round(mrr / n, 3), n


Q = json.load(open("data/eval_questions.json", encoding="utf-8"))
methods = [
    ("hybrid",        lambda q, k: hybrid.hybrid_retrieve(q, top_k=k)),
    ("hybrid+rerank", lambda q, k: rerank.retrieve(q, top_k=k, pool=20)),
]
print(f"{'method':16} {'hit@1':>8} {'hit@5':>8} {'MRR':>7}")
for name, fn in methods:
    h1, h5, mrr, n = score(fn, Q)
    print(f"{name:16} {h1:>3}/{n:<4} {h5:>3}/{n:<4} {mrr:>7}")