
import os
import sys
import time
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  

DSN = "postgresql://postgres:mygov@localhost:5433/mygov"
# Pin a real model via OPENROUTER_MODEL for reliable answers + deterministic
# rewrites; the "openrouter/free" default is a flaky auto-router (varies per call,
# so temp=0 isn't truly deterministic and low-quality turns pollute chat history).
MODEL_LLM = os.getenv("OPENROUTER_MODEL", "openrouter/free")
TOP_K = 5

_embedder = SentenceTransformer("BAAI/bge-m3")
_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

SYSTEM_PROMPT = """Sen my.gov.uz davlat xizmatlari bo'yicha yordamchisan.

QAT'IY QOIDALAR:
- Faqat quyidagi KONTEKSTdagi ma'lumotga asoslanib javob ber.
- Agar javob kontekstda bo'lmasa, aniq shu jumla bilan javob ber: "Menda bu haqda ishonchli ma'lumot yo'q." Hech narsani o'ylab topma.
- Har doim qaysi xizmatga asoslanganingni ko'rsat: xizmat nomi va havolasi (URL).
- Narx, muddat, hujjatlar haqida faqat kontekstda yozilganini ayt - taxmin qilma.
- Foydalanuvchi tilida javob ber (o'zbek/rus). Boshqa hech qanday belgi yoki teg chiqarma."""


def retrieve(question, k=TOP_K):
    qv = _embedder.encode(question, normalize_embeddings=True)
    conn = psycopg2.connect(DSN)
    register_vector(conn)
    cur = conn.cursor()
    cur.execute("""
        SELECT service_id, title, url, text, keywords, 1 - (embedding <=> %s) AS score
        FROM chunks
        ORDER BY embedding <=> %sif
        LIMIT %s;
    """, (qv, qv, k))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows  # (service_id, title, url, text, keywords, score) — matches build_context


def build_context(rows):
    blocks = []
    for i, (sid, title, url, text, keywords, score) in enumerate(rows, 1):
        blocks.append(f"[{i}] Xizmat: {title}\nURL: {url}\nMatn: {text}")
    return "\n\n".join(blocks)


def answer(question, k=TOP_K):
    rows = retrieve(question, k)
    user_msg = f"KONTEKST:\n{build_context(rows)}\n\nSAVOL: {question}"
    last = None
    for attempt in range(4):
        try:
            resp = _client.chat.completions.create(
                model=MODEL_LLM,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.2,
            )
            return resp.choices[0].message.content, rows
        except Exception as e:
            last = e
            time.sleep(2 * (attempt + 1))  # 2s, 4s, 6s backoff
    raise last


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "ID karta yo'qolsa nima qilaman?"
    text, rows = answer(q)
    print(f"\nSAVOL: {q}\n" + "=" * 60)
    print(text)
    print("\n" + "-" * 60 + "\nRETRIEVED:")
    for sid, title, url, _, _, score in rows:
        print(f"  [{score:.3f}] #{sid} {title}")