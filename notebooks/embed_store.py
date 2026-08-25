"""
Phase 1, steps 2-3 — embed chunks with bge-m3, store in Postgres/pgvector.

Prereqs:
  - Postgres+pgvector running (see docker command).
  - pip install sentence-transformers psycopg2-binary pgvector
  - data/chunks.jsonl exists (from chunk_services.py)

Run:
  python embed_store.py

First run downloads bge-m3 (~2.2 GB). It auto-uses your GPU if CUDA is present.
Re-running is safe: it TRUNCATEs and reloads.
"""
import json
import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

DSN = "postgresql://postgres:mygov@localhost:5433/mygov"
MODEL = "BAAI/bge-m3"
CHUNKS = "data/chunks.jsonl"
DIM = 1024         
BATCH = 12         


def main():
    rows = [json.loads(l) for l in open(CHUNKS, encoding="utf-8")]
    print(f"{len(rows)} chunks to embed")

    model = SentenceTransformer(MODEL)
    model.max_seq_length = 512 
    print("device:", model.device)

    # embed_text = title + chunk, so tiny chunks still know their service
    vectors = model.encode(
        [r["embed_text"] for r in rows],
        batch_size=BATCH,
        normalize_embeddings=True,   # -> use cosine distance in SQL
        show_progress_bar=True,
    )  # shape (N, 1024)

    conn = psycopg2.connect(DSN)
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()
    register_vector(conn)  # lets us pass numpy arrays straight in

    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id    TEXT PRIMARY KEY,
            service_id  INTEGER NOT NULL,
            title       TEXT,
            url         TEXT,
            lang        TEXT,
            chunk_index INTEGER,
            text        TEXT NOT NULL,
            keywords    TEXT,
            embedding   vector({DIM})
        );
    """)
    cur.execute("TRUNCATE chunks;")
    conn.commit()

    execute_values(
        cur,
        "INSERT INTO chunks (chunk_id, service_id, title, url, lang, "
        "chunk_index, text, keywords, embedding) VALUES %s",
        [(r["chunk_id"], r["service_id"], r["title"], r["url"], r["lang"],
          r["chunk_index"], r["text"], r.get("keywords", ""), v)
         for r, v in zip(rows, vectors)],
    )
    conn.commit()

    # ANN index. At 926 rows a flat scan is already instant, so this barely
    # matters now — but it's the right habit and matters at 100k+ rows.
    cur.execute("CREATE INDEX IF NOT EXISTS chunks_emb_idx ON chunks "
                "USING hnsw (embedding vector_cosine_ops);")
    conn.commit()

    # Two more DB objects the hybrid path needs are built downstream, not here:
    #   - the `fts` tsvector column + GIN index: created lazily by
    #     hybrid.ensure_fts() on the first hybrid query.
    #   - the `vocab` trigram table (typo-fix): built by `python build_vocab.py`.
    # Run build_vocab.py after this script.

    cur.execute("SELECT COUNT(*) FROM chunks;")
    print("stored rows:", cur.fetchone()[0])
    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
