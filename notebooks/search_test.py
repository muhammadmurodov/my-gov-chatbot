"""
Retrieval sanity check — embed a question, pull the nearest chunks.
    python search_test.py "tonirovka ruxsatnomasi qanday olinadi?"
"""
import sys
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

DSN = "postgresql://postgres:mygov@localhost:5433/mygov"
model = SentenceTransformer("BAAI/bge-m3")

query = sys.argv[1] if len(sys.argv) > 1 else "chet elga chiqish uchun ruxsat"
qv = model.encode(query, normalize_embeddings=True)

conn = psycopg2.connect(DSN)
register_vector(conn)
cur = conn.cursor()
# 1 - cosine distance = similarity (higher is better). <=> is pgvector's cosine op.
cur.execute("""
    SELECT title, service_id, 1 - (embedding <=> %s) AS score, text
    FROM chunks
    ORDER BY embedding <=> %s
    LIMIT 5;
""", (qv, qv))

print(f"\nQUERY: {query}\n" + "=" * 60)
for title, sid, score, text in cur.fetchall():
    print(f"[{score:.3f}] #{sid}  {title}")
    print(f"        {text[:110].strip()}...\n")
cur.close()
conn.close()
