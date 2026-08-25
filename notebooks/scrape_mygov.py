"""
my.gov.uz service catalog scraper (resumable, two-pass).

Pass 1: walk all spheres -> manifest of every service (data/manifest.jsonl)
Pass 2: fetch each service's full passport -> data/raw/{id}.json (skips existing)

Run again after a crash: it skips whatever is already on disk.

    python scrape_mygov.py
"""
import json
import time
import random
from pathlib import Path
import httpx

BASE = "https://my.gov.uz/api/static/services/my-gov/index"
DETAIL = "https://my.gov.uz/api/static/servicePassport/my-gov/view"

# All 25 sphere IDs, pulled from the homepage.
SPHERE_IDS = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 15, 16, 17, 18,
              22, 23, 29, 30, 32, 34, 37, 38, 40, 42]

LANG = "uz"  # also worth a second run with "ru"
DATA = Path("data")
RAW = DATA / "raw"
MANIFEST = DATA / "manifest.jsonl"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (research; contact: your-email@example.com)",
    "Accept": "application/json",
    "Accept-Language": "uz,ru;q=0.8,en;q=0.6",
}


def polite_sleep():
    time.sleep(1.0 + random.random())  # 1-2s, jittered


def get_json(client, url, params):
    r = client.get(url, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def pass1_manifest(client):
    """Collect every service across all spheres into manifest.jsonl."""
    if MANIFEST.exists():
        print(f"Manifest exists ({sum(1 for _ in MANIFEST.open())} rows) — skipping pass 1.")
        return
    seen = set()
    with MANIFEST.open("w", encoding="utf-8") as f:
        for sid in SPHERE_IDS:
            data = get_json(client, BASE, {"serviceSphereId": sid, "lang": LANG})
            services = data.get("result", {}).get("models", data.get("result", []))
            for s in services:
                if s["id"] in seen:
                    continue
                seen.add(s["id"])
                f.write(json.dumps({
                    "id": s["id"],
                    "sphere_id": sid,
                    "title": s.get("title"),
                    "short_title": s.get("short_title"),
                    "code": s.get("code"),
                    "is_payable": s.get("is_payable_service"),
                    "keywords": s.get("keywords", []),
                }, ensure_ascii=False) + "\n")
            print(f"sphere {sid}: {len(services)} services (total {len(seen)})")
            polite_sleep()
    print(f"Pass 1 done: {len(seen)} unique services.")


def pass2_details(client):
    """Fetch full passport for each service; skip whatever is already saved."""
    RAW.mkdir(parents=True, exist_ok=True)
    ids = [json.loads(l)["id"] for l in MANIFEST.open(encoding="utf-8")]
    for i, sid in enumerate(ids, 1):
        out = RAW / f"{sid}.json"
        if out.exists():
            continue  # <- resumability
        try:
            data = get_json(client, DETAIL, {"service_id": sid, "lang": LANG})
            out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[{i}/{len(ids)}] saved {sid}")
        except httpx.HTTPStatusError as e:
            print(f"[{i}/{len(ids)}] FAILED {sid}: {e.response.status_code}")
        polite_sleep()
    print(f"Pass 2 done: {len(list(RAW.glob('*.json')))} details on disk.")


def main():
    DATA.mkdir(exist_ok=True)
    with httpx.Client() as client:
        pass1_manifest(client)
        pass2_details(client)


if __name__ == "__main__":
    main()
