"""
Phase 2 - hybrid retrieval: vector + full-text, fused with RRF.
  python hybrid.py "kadastr pasporti qanday olinadi?"
"""
import sys
import time
import psycopg2
import psycopg2.pool
from pgvector.psycopg2 import register_vector
import rag  # reuse embedder, DSN, LLM client, prompt
import re

# One shared connection pool instead of connect()/close() per request. Under
# Open WebUI's default streaming each turn hits the DB, and uvicorn runs sync
# endpoints across a threadpool — so use the threaded pool. Created lazily so
# importing this module never needs a live DB (tests import it without one).
_CONN_POOL = None


def _get_conn_pool():
    global _CONN_POOL
    if _CONN_POOL is None:
        _CONN_POOL = psycopg2.pool.ThreadedConnectionPool(1, 10, rag.DSN)
    return _CONN_POOL

def correct_query(cur, question, min_sim=0.45, min_len=4):
    """Replace out-of-vocabulary tokens with the closest real corpus word."""
    fixed, changed = [], []
    for tok in re.findall(r"\w+", question.lower()):
        if len(tok) < min_len:
            fixed.append(tok)
            continue
        cur.execute("SELECT 1 FROM vocab WHERE word = %s", (tok,))
        if cur.fetchone():                    # known word, leave alone
            fixed.append(tok)
            continue
        cur.execute("""
            SELECT word, similarity(word, %s) AS s
            FROM vocab
            WHERE word %% %s                  -- trigram index scan
            ORDER BY s DESC, ndoc DESC        -- tie-break toward common words
            LIMIT 1
        """, (tok, tok))
        r = cur.fetchone()
        if r and r[1] >= min_sim:
            fixed.append(r[0])
            changed.append((tok, r[0]))
        else:
            fixed.append(tok)                 # no confident fix, keep original
    return " ".join(fixed), changed


def ensure_fts(cur):
    cur.execute("""
        ALTER TABLE chunks ADD COLUMN IF NOT EXISTS fts tsvector
        GENERATED ALWAYS AS
        (to_tsvector('simple', coalesce(title,'') || ' ' || coalesce(keywords,'') || ' ' || coalesce(text,''))) STORED;
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS chunks_fts_idx ON chunks USING gin(fts);")


def vector_ids(cur, qv, n=20):        # nearest by MEANING
    cur.execute("SELECT chunk_id FROM chunks ORDER BY embedding <=> %s LIMIT %s", (qv, n))
    return [r[0] for r in cur.fetchall()]


def fts_ids(cur, query, n=20):        # nearest by EXACT WORDS
    cur.execute("""
        SELECT chunk_id FROM chunks
        WHERE fts @@ plainto_tsquery('simple', %s)
        ORDER BY ts_rank(fts, plainto_tsquery('simple', %s)) DESC
        LIMIT %s
    """, (query, query, n))
    return [r[0] for r in cur.fetchall()]


def rrf(*ranked_lists, k=60):
    # each list votes 1/(k+rank) per item; return items sorted by summed votes
    scores = {}
    for lst in ranked_lists:
        for rank, cid in enumerate(lst):
            scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def hybrid_retrieve(question, top_k=5, pool=20):
    conn_pool = _get_conn_pool()
    conn = conn_pool.getconn()
    try:
        register_vector(conn)  # idempotent per connection; sets up the vector adapter
        cur = conn.cursor()
        ensure_fts(cur)
        conn.commit()

        q_fts, changed = correct_query(cur, question)
        if changed:
            print(f"  [typo-fix] {changed}")

        qv = rag._embedder.encode(question, normalize_embeddings=True)  # ORIGINAL
        fused = rrf(vector_ids(cur, qv, pool), fts_ids(cur, q_fts, pool))[:top_k]
        rows = []
        for cid, score in fused:
            cur.execute("SELECT service_id, title, url, text, keywords FROM chunks WHERE chunk_id=%s", (cid,))
            sid, title, url, text, keywords = cur.fetchone()
            rows.append((sid, title, url, text, keywords, score))
        cur.close()
        conn.commit()
        return rows
    except Exception:
        conn.rollback()  # leave the pooled connection clean, not mid-aborted-txn
        raise
    finally:
        conn_pool.putconn(conn)


def answer(question, k=5):
    rows = hybrid_retrieve(question, k)
    user_msg = f"KONTEKST:\n{rag.build_context(rows)}\n\nSAVOL: {question}"
    last = None
    for attempt in range(4):
        try:
            resp = rag._client.chat.completions.create(
                model=rag.MODEL_LLM,
                messages=[{"role": "system", "content": rag.SYSTEM_PROMPT},
                          {"role": "user", "content": user_msg}],
                temperature=0.2,
            )
            return resp.choices[0].message.content, rows
        except Exception as e:
            last = e
            time.sleep(2 * (attempt + 1))
    raise last


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "kadastr pasporti qanday olinadi?"
    print(f"\nQUERY: {q}\n" + "=" * 60)
    for sid, title, url, _, _, score in hybrid_retrieve(q):
        print(f"  [{score:.4f}] #{sid}  {title}")