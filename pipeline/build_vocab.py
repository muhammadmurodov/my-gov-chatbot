"""
Build the `vocab` table that powers the query typo-fix in hybrid.correct_query.

For every out-of-vocabulary query token, hybrid.py finds the closest real corpus
word via a pg_trgm trigram similarity scan over this table. So `vocab` must hold
every word that appears in the corpus, plus how many chunks each occurs in (`ndoc`,
used only as a tie-break toward common words).

Schema (matches what hybrid.py queries):
    vocab(word TEXT PRIMARY KEY, ndoc INTEGER)   + GIN trigram index on word
    requires the pg_trgm extension (for `%` operator and similarity()).

Prereq: chunks table already populated (run embed_store.py first).
Run:
    python build_vocab.py

Idempotent: drops and rebuilds vocab each run.
"""
import re
from collections import Counter

import psycopg2

DSN = "postgresql://postgres:mygov@localhost:5433/mygov"
TOKEN = re.compile(r"\w+", re.UNICODE)


def corpus_tokens_per_chunk(cur):
    """Yield the set of distinct lowercased tokens for each chunk.

    Words are drawn from title + keywords + text — the same fields the FTS column
    indexes — so the typo-fix vocabulary matches what retrieval actually searches.
    """
    cur.execute("SELECT coalesce(title,''), coalesce(keywords,''), coalesce(text,'') FROM chunks")
    for title, keywords, text in cur.fetchall():
        blob = f"{title} {keywords} {text}".lower()
        yield set(TOKEN.findall(blob))


def main():
    conn = psycopg2.connect(DSN)
    cur = conn.cursor()

    # document frequency: in how many chunks does each word appear
    ndoc = Counter()
    for tokens in corpus_tokens_per_chunk(cur):
        ndoc.update(tokens)
    print(f"{len(ndoc)} distinct words across the corpus")

    cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
    cur.execute("DROP TABLE IF EXISTS vocab;")
    cur.execute("CREATE TABLE vocab (word TEXT PRIMARY KEY, ndoc INTEGER NOT NULL);")

    from psycopg2.extras import execute_values
    execute_values(cur, "INSERT INTO vocab (word, ndoc) VALUES %s",
                   list(ndoc.items()))

    # trigram index: makes `WHERE word %% %s` (fuzzy match) an index scan
    cur.execute("CREATE INDEX IF NOT EXISTS vocab_trgm_idx ON vocab USING gin (word gin_trgm_ops);")
    conn.commit()

    cur.execute("SELECT count(*) FROM vocab;")
    print("vocab rows:", cur.fetchone()[0])
    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
