"""
Phase 2 - rebuild chunks.jsonl WITH synonyms so tonirovka/propiska become findable.

Adds each service's synonyms to:
  - embed_text  -> the vector search can now "see" the loanwords
  - keywords    -> a new field the FTS column will index

  python chunk_with_synonyms.py data/services.jsonl data/synonyms.jsonl data/chunks.jsonl
"""
import json
import re
import sys

TARGET_CHARS, OVERLAP_CHARS, SNAP_WINDOW = 2000, 200, 400


def chunk_text(text, target=TARGET_CHARS, overlap=OVERLAP_CHARS):
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if len(text) <= target:
        return [text] if text else []
    chunks, start = [], 0
    while start < len(text):
        end = start + target
        if end >= len(text):
            chunks.append(text[start:].strip()); break
        window = text[start:end]
        bounds = list(re.finditer(r"[.!?]\s", window))
        if bounds and bounds[-1].end() > target - SNAP_WINDOW:
            cut = start + bounds[-1].end()
        else:
            sp = text.rfind(" ", start + target - SNAP_WINDOW, end)
            cut = sp if sp > start else end
        chunks.append(text[start:cut].strip())
        start = max(cut - overlap, start + 1)
    return [c for c in chunks if c]


def main(inp, syn_path, outp):
    synonyms = json.load(open(syn_path, encoding="utf-8"))  # {"415": [...], ...}
    n_services = n_chunks = enriched = 0
    with open(inp, encoding="utf-8") as f, open(outp, "w", encoding="utf-8") as out:
        for line in f:
            r = json.loads(line)
            n_services += 1
            kws = synonyms.get(str(r["id"]), [])            # synonyms for THIS service
            if kws:
                enriched += 1
            kw_str = " ".join(kws)
            pieces = chunk_text(r.get("content", ""))
            title = r.get("title", "")
            for i, piece in enumerate(pieces):
                n_chunks += 1
                # keywords go into embed_text (vectors) AND their own field (FTS)
                embed_text = f"{title}\n{kw_str}\n{piece}" if kw_str else f"{title}\n{piece}"
                out.write(json.dumps({
                    "chunk_id": f'{r["id"]}-{i}', "service_id": r["id"],
                    "title": title, "url": r.get("url"), "lang": r.get("lang", "uz"),
                    "chunk_index": i, "n_chunks": len(pieces),
                    "text": piece, "keywords": kw_str, "embed_text": embed_text,
                }, ensure_ascii=False) + "\n")
    print(f"{n_services} services ({enriched} enriched with synonyms) -> {n_chunks} chunks -> {outp}")


if __name__ == "__main__":
    inp = sys.argv[1] if len(sys.argv) > 1 else "data/services.jsonl"
    syn = sys.argv[2] if len(sys.argv) > 2 else "data/synonyms.jsonl"
    outp = sys.argv[3] if len(sys.argv) > 3 else "data/chunks.jsonl"
    main(inp, syn, outp)