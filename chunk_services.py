"""
Phase 1, step 1 — chunk services into retrieval units.

Input : services.jsonl   (one service per line: id, url, title, description, content)
Output: chunks.jsonl      (one chunk per line, each carrying title + url metadata)

Design (defensible in interview):
- Target ~500 tokens/chunk. Most services fit in ONE chunk, so chunking barely
  fires — but long services (steps, legal basis) still split cleanly.
- Break at sentence boundaries near the target, not mid-word.
- Small overlap so a fact split across a boundary isn't lost.
- Every chunk stores service_id, title, url, chunk_index — needed for citations.
- `embed_text` prepends the title so short chunks still have context when embedded.

    python chunk_services.py data/services.jsonl data/chunks.jsonl
"""
import json
import re
import sys

# ~4 chars/token for Uzbek Latin. Swap in the bge-m3 tokenizer later for exactness.
TARGET_CHARS = 2000   # ~500 tokens
OVERLAP_CHARS = 200   # ~50 tokens
SNAP_WINDOW = 400     # look this far back for a sentence end before hard-cutting


def chunk_text(text, target=TARGET_CHARS, overlap=OVERLAP_CHARS):
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if len(text) <= target:
        return [text] if text else []
    chunks, start = [], 0
    while start < len(text):
        end = start + target
        if end >= len(text):
            chunks.append(text[start:].strip())
            break
        window = text[start:end]
        # prefer a sentence boundary in the last SNAP_WINDOW chars of the window
        bounds = list(re.finditer(r"[.!?]\s", window))
        if bounds and bounds[-1].end() > target - SNAP_WINDOW:
            cut = start + bounds[-1].end()
        else:  # else fall back to the nearest space
            sp = text.rfind(" ", start + target - SNAP_WINDOW, end)
            cut = sp if sp > start else end
        chunks.append(text[start:cut].strip())
        start = max(cut - overlap, start + 1)
    return [c for c in chunks if c]


def main(inp, outp):
    n_services = n_chunks = 0
    with open(inp, encoding="utf-8") as f, open(outp, "w", encoding="utf-8") as out:
        for line in f:
            r = json.loads(line)
            pieces = chunk_text(r.get("content", ""))
            n_services += 1
            for i, piece in enumerate(pieces):
                n_chunks += 1
                title = r.get("title", "")
                out.write(json.dumps({
                    "chunk_id": f'{r["id"]}-{i}',
                    "service_id": r["id"],
                    "title": title,
                    "url": r.get("url"),
                    "lang": r.get("lang", "uz"),
                    "chunk_index": i,
                    "n_chunks": len(pieces),
                    "text": piece,
                    "embed_text": f"{title}\n{piece}" if title else piece,
                }, ensure_ascii=False) + "\n")
    print(f"{n_services} services -> {n_chunks} chunks -> {outp}")


if __name__ == "__main__":
    inp = sys.argv[1] if len(sys.argv) > 1 else "data/services.jsonl"
    outp = sys.argv[2] if len(sys.argv) > 2 else "data/chunks.jsonl"
    main(inp, outp)