"""
Offline threshold calibration for the two refusal gates.

Runs each labeled question through the SAME scoring the live gates use — the topic
classifier probability (drives CLF_OFFTOPIC) and the top reranker score (drives
GATE_FLOOR) — WITHOUT ever calling the answer LLM, so it is cheap and deterministic.
It then sweeps both thresholds and reports how each choice trades a false-accept
(letting a query that should be refused through) against a false-reject (refusing a
query that should be answered), so you pick the thresholds from data, not by feel.

Input CSV columns: `question,label`. Labels:
  in_domain            answerable and in the corpus       -> should be ANSWERED
  off_topic            not a my.gov.uz topic at all       -> should be refused by the CLASSIFIER
  in_domain_no_answer  gov-ish, but not in the corpus     -> should be refused by the FLOOR
  ambiguous            borderline; scored + printed, but excluded from threshold picks

The GPU is usually held by the running API, so run this on CPU:

  CUDA_VISIBLE_DEVICES="" HF_HOME=/mnt/NewData/hf-cache HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 ./mygov/bin/python eval/calibrate.py eval/labeled_seed.csv

Optional:  --scored-out scored.csv   (dump per-question clf_prob/top_score for inspection)
"""
import argparse
import csv
import os
import re
import sys

SRC = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, SRC)


def score_question(q):
    """(clf_prob, top_reranker_score) for one question — mirrors rerank.answer's gates
    (classifier on the query, then hybrid+rerank), minus the answer LLM. No history, so
    the query is its own 'standalone'. top score is 0.0 when retrieval returns nothing.

    Imports the heavy modules lazily so --from-scored can re-sweep with no GPU/DB."""
    import rag
    import rerank
    emb = rag._embedder.encode(q, normalize_embeddings=True)
    p = float(rerank._clf.predict_proba([emb])[0][1])
    rows = rerank.retrieve(q)
    top = float(rows[0][-1]) if rows else 0.0
    return p, top


def current_thresholds():
    """Read CLF_OFFTOPIC / GATE_FLOOR from src/rerank.py by text, so showing the current
    config never triggers a model load (relevant in --from-scored mode)."""
    src = open(os.path.join(SRC, "rerank.py"), encoding="utf-8").read()
    grab = lambda name: float(re.search(rf"^{name}\s*=\s*([0-9.]+)", src, re.M).group(1))
    return grab("CLF_OFFTOPIC"), grab("GATE_FLOOR")


# The live gate answers only when BOTH gates pass, so calibration is judged on that same
# end-to-end OUTCOME, not per-gate. Outcome labels:
#   in_domain            -> should be ANSWERED  (a "positive")
#   off_topic            -> should be REFUSED    (a "negative")
#   in_domain_no_answer  -> should be REFUSED    (a "negative" — by EITHER gate; both are fine)
# ambiguous is judgment; it's reported but never counts toward an error.
POSITIVE = {"in_domain"}
NEGATIVE = {"off_topic", "in_domain_no_answer"}


def load(path):
    with open(path, encoding="utf-8") as f:
        return [(r["question"].strip(), r["label"].strip())
                for r in csv.DictReader(f) if r.get("question", "").strip()]


def load_scored(path):
    with open(path, encoding="utf-8") as f:
        return [(r["question"].strip(), r["label"].strip(),
                 float(r["clf_prob"]), float(r["top_score"]))
                for r in csv.DictReader(f) if r.get("question", "").strip()]


def outcome_errors(scored, t_clf, t_floor):
    """(false_reject, false_accept) for one (t_clf, t_floor) pair, judged end-to-end:
    the pipeline answers iff clf_prob >= t_clf AND top_score >= t_floor."""
    fr = fa = 0
    for _q, label, p, top in scored:
        answered = (p >= t_clf and top >= t_floor)
        if label in POSITIVE and not answered:
            fr += 1
        elif label in NEGATIVE and answered:
            fa += 1
    return fr, fa


def best_pair(scored, clf_grid, floor_grid):
    """The (t_clf, t_floor) with fewest total errors; ties break toward the LEAST
    aggressive gate (lowest thresholds), since a false-reject usually costs more than a
    false-accept that the LLM's own grounded-answer prompt can still catch downstream."""
    best = None
    for tc in clf_grid:
        for tf in floor_grid:
            fr, fa = outcome_errors(scored, tc, tf)
            key = (fr + fa, tc, tf)              # min errors, then smallest thresholds
            if best is None or key < best[0]:
                best = (key, (tc, tf), (fr, fa))
    return best[1], best[2]


def summarize(scored):
    by = {}
    for row in scored:
        by.setdefault(row[1], []).append(row)
    for lab in ("in_domain", "off_topic", "in_domain_no_answer", "ambiguous"):
        g = by.get(lab, [])
        if not g:
            continue
        p = sorted(s[2] for s in g)
        t = sorted(s[3] for s in g)
        print(f"  {lab:<20} n={len(g):>2}  "
              f"clf[min/med/max]={p[0]:.2f}/{p[len(p)//2]:.2f}/{p[-1]:.2f}  "
              f"top[min/med/max]={t[0]:.2f}/{t[len(t)//2]:.2f}/{t[-1]:.2f}")

    pos = [s for s in scored if s[1] in POSITIVE]
    off = [s for s in scored if s[1] == "off_topic"]
    if pos and off:
        gap_lo, gap_hi = max(s[2] for s in off), min(s[2] for s in pos)
        verdict = "clean gap" if gap_hi > gap_lo else "OVERLAP"
        print(f"\n  classifier separability: off_topic max clf={gap_lo:.2f} vs "
              f"in_domain min clf={gap_hi:.2f}  -> {verdict}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="labeled questions (question,label), or a scored CSV with --from-scored")
    ap.add_argument("--scored-out", help="write per-question scores here")
    ap.add_argument("--from-scored", action="store_true",
                    help="read clf_prob/top_score columns from CSV and skip model scoring "
                         "(re-sweep instantly, no GPU/DB needed)")
    args = ap.parse_args()

    if args.from_scored:
        scored = load_scored(args.csv)
        print(f"Re-sweeping {len(scored)} pre-scored questions (no models called).")
    else:
        data = load(args.csv)
        print(f"Scoring {len(data)} questions through classifier + hybrid + reranker "
              f"(no answer LLM)...", flush=True)
        scored = []
        for i, (q, label) in enumerate(data, 1):
            p, top = score_question(q)
            scored.append((q, label, p, top))
            print(f"  [{i:>3}/{len(data)}] p={p:.3f} top={top:.3f}  {label:<20} {q[:48]}", flush=True)

    if args.scored_out:
        with open(args.scored_out, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["question", "label", "clf_prob", "top_score"])
            w.writerows((q, l, f"{p:.4f}", f"{t:.4f}") for q, l, p, t in scored)
        print(f"Wrote per-question scores -> {args.scored_out}")

    n_pos = sum(s[1] in POSITIVE for s in scored)
    n_neg = sum(s[1] in NEGATIVE for s in scored)
    print(f"\nScore distribution by label ({n_pos} answer-worthy, {n_neg} refuse-worthy):")
    summarize(scored)

    clf_grid = [round(x / 100, 2) for x in range(5, 75, 5)]
    floor_grid = [round(x / 100, 2) for x in range(0, 55, 5)]

    cur = current_thresholds()
    cur_fr, cur_fa = outcome_errors(scored, *cur)
    (bc, bf), (bfr, bfa) = best_pair(scored, clf_grid, floor_grid)

    print("\n" + "=" * 66)
    print(f"  {'':<10}{'CLF_OFFTOPIC':>14}{'GATE_FLOOR':>12}"
          f"{'false-reject':>14}{'false-accept':>14}")
    print(f"  {'current':<10}{cur[0]:>14.2f}{cur[1]:>12.2f}{cur_fr:>14}{cur_fa:>14}")
    print(f"  {'suggest':<10}{bc:>14.2f}{bf:>12.2f}{bfr:>14}{bfa:>14}")
    print("=" * 66)
    print("false-reject = an answer-worthy question refused;  false-accept = a refuse-worthy")
    print("question answered. Suggestion minimizes their sum on THIS set — treat a small seed")
    print("as directional; grow it from MYGOV_QUERY_LOG traffic before trusting the numbers.")


if __name__ == "__main__":
    main()
