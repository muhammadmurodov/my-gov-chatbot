"""
my.gov.uz public service catalog scraper (for RAG).

Scrapes the PUBLIC, server-rendered detail pages at https://my.gov.uz/uz/service/{id}
— the same HTML any browser receives. No private/signed API is touched.

Source of IDs: the official sitemap https://my.gov.uz/sitemap/uz-service.xml

Output:
  data/services/{id}.json   one structured record per service (resumable — skips existing)
  data/services.jsonl       all records concatenated, ready for chunking/embedding

Resumable: rerun after a crash and it skips whatever is already saved.

    python scrape_services.py
"""
import json
import re
import time
import random
import html as H
from pathlib import Path
import httpx

LANG = "uz"
SITEMAP = f"https://my.gov.uz/sitemap/{LANG}-service.xml"
PAGE = "https://my.gov.uz/{lang}/service/{id}"

DATA = Path("data")
OUT = DATA / "services"
COMBINED = DATA / "services.jsonl"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (research RAG dataset build)",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "uz,ru;q=0.8,en;q=0.6",
}

# Footer begins here on every page; content above it is the passport.
FOOTER_MARKERS = [
    "sizga ma'qul keldimi",
    "sizga ma’qul keldimi",
    "Diqqat! Agar matnda xato",
]


def polite_sleep():
    time.sleep(1.0 + random.random())  # 1-2s, jittered


def visible_text(raw):
    body = re.sub(r"<script.*?</script>", " ", raw, flags=re.S)
    body = re.sub(r"<style.*?</style>", " ", body, flags=re.S)
    text = re.sub(r"<[^>]+>", " ", body)
    text = H.unescape(text)              # decode &lt;br&gt; etc.
    text = re.sub(r"<[^>]+>", " ", text)  # sweep tags the decode revealed
    return re.sub(r"\s+", " ", text).strip()


def meta(raw, name, attr="name"):
    m = re.search(rf'{attr}="{re.escape(name)}"\s+content="([^"]*)"', raw)
    return H.unescape(m.group(1)).strip() if m else None


def og_stats(raw):
    """Pull the small summary block SvelteKit inlines for the OG image."""
    stats = {}
    labels = {
        "avg_rating": r"O'rtacha baho",
        "form_time_min": r"Ariza to'ldirish vaqti",
        "applications": r"Arizalar soni",
        "comments": r"Sharhlar soni",
    }
    for key, lbl in labels.items():
        m = re.search(rf'"{lbl}",([\d.]+)', raw)
        if m:
            v = m.group(1)
            stats[key] = float(v) if "." in v else int(v)
    return stats


def extract(raw, sid):
    title = meta(raw, "og:title", attr="property") or meta(raw, "title", attr="property")
    if not title:
        mt = re.search(r"<title>(.*?)</title>", raw, re.S)
        title = H.unescape(mt.group(1)).strip() if mt else None
    desc = meta(raw, "description") or meta(raw, "og:description", attr="property")

    text = visible_text(raw)

    # Trim footer.
    cut = len(text)
    for mk in FOOTER_MARKERS:
        i = text.find(mk)
        if i != -1:
            cut = min(cut, i)
    text = text[:cut]

    # Trim header/nav chrome. The nav ends with the twin login buttons
    # ("Kirish Kirish"); the H1 and passport follow immediately after.
    anchor = "Kirish Kirish"
    j = text.find(anchor)
    if j != -1:
        text = text[j + len(anchor):]
    elif title:
        # fallback: start at the H1 (2nd occurrence of the title text)
        first = text.find(title)
        second = text.find(title, first + len(title)) if first != -1 else -1
        if second != -1:
            text = text[second:]
    content = text.strip()

    return {
        "id": sid,
        "url": PAGE.format(lang=LANG, id=sid),
        "lang": LANG,
        "title": title,
        "description": desc,
        "stats": og_stats(raw),
        "content": content,
    }


def get_ids(client):
    r = client.get(SITEMAP, headers=HEADERS, timeout=60)
    r.raise_for_status()
    ids = sorted({int(m) for m in re.findall(r"/service/(\d+)", r.text)})
    return ids


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with httpx.Client(follow_redirects=True) as client:
        ids = get_ids(client)
        print(f"{len(ids)} services in sitemap.")
        done = 0
        for i, sid in enumerate(ids, 1):
            out = OUT / f"{sid}.json"
            if out.exists():
                done += 1
                continue
            try:
                r = client.get(PAGE.format(lang=LANG, id=sid), headers=HEADERS, timeout=45)
                r.raise_for_status()
                rec = extract(r.text, sid)
                out.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
                done += 1
                clen = len(rec["content"])
                print(f"[{i}/{len(ids)}] {sid}: {clen} chars — {(rec['title'] or '')[:60]}")
            except httpx.HTTPStatusError as e:
                print(f"[{i}/{len(ids)}] {sid}: FAILED {e.response.status_code}")
            except Exception as e:
                print(f"[{i}/{len(ids)}] {sid}: ERROR {e}")
            polite_sleep()

    # Rebuild the combined RAG file from everything on disk.
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(OUT.glob("*.json"), key=lambda p: int(p.stem))]
    with COMBINED.open("w", encoding="utf-8") as f:
        for rec in recs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Done: {len(recs)} services saved. Combined -> {COMBINED}")


if __name__ == "__main__":
    main()
