"""Step 1: draft English seed Q&A from the raw corpus with an Adaption hosted model.

    python pipeline/step1_build_seed.py --dry-run      # count units, estimate credits, no API calls
    python pipeline/step1_build_seed.py --limit 20     # small paid test (fractions of a credit)
    python pipeline/step1_build_seed.py                # full run (resumable; results are cached)
    python pipeline/step1_build_seed.py --assemble     # rebuild output files from the cache only

Writes data/seed/train_en.jsonl, eval_en.jsonl and eval_real_en.jsonl.
"""
import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from behavior_rows import behavior_rows
from common import get_client, read_jsonl, write_jsonl
from config import (BUNDLE_CHARS, CATEGORIES, CHARS_PER_QUESTION, EXCLUDE_TYPES,
                    GEN_MODEL, GEN_PRICE_PER_M, HOLDOUT_SHARE, MAX_QUESTIONS_PER_UNIT,
                    MIN_CHUNK_CHARS, KNOWLEDGE, RAW, SEED, WORK)

CACHE = WORK / "seed_cache"
SHORT_CHUNK = 800  # chunks below this are bundled with neighbours of the same type

INSTRUCTIONS = f"""You write training data for the support bot in Adaption Labs' Discord server.
From the SOURCE text, write {{n}} question-and-answer pairs.

Questions:
- Things a real community member would ask in a support ticket: varied wording, mostly casual,
  some short, some describing what they're building. Don't start them all with "What".
- Each must be answerable from the SOURCE alone.

Answers (written as the support bot):
- Use only facts in the SOURCE. Concise: 2 to 6 sentences, or a short code block for API questions.
- If a fact is tied to a date (event dates, deadlines, launches), include the date.
- End with the most relevant source URL from the [Source: ...] headers.

Never include names or handles of private individuals. If the SOURCE has nothing a user would
need support with (pure hype, greetings, retweets), return [].

Return ONLY a JSON array:
[{{"question": "...", "answer": "...", "category": "<one of {CATEGORIES}>", "source_url": "..."}}]"""


def body_len(text: str) -> int:
    return len(text.split("\n", 1)[1]) if "\n" in text else 0


def build_units():
    chunks = read_jsonl(KNOWLEDGE / "corpus_chunks.jsonl")
    kept = [c for c in chunks if c["type"] not in EXCLUDE_TYPES and body_len(c["text"]) >= 40]
    long_units, short_by_type = [], defaultdict(list)
    for c in kept:
        if len(c["text"]) >= SHORT_CHUNK:
            long_units.append([c])
        else:
            short_by_type[c["type"]].append(c)
    bundles = []
    for items in short_by_type.values():
        items.sort(key=lambda c: c.get("date_published") or "")
        cur, size = [], 0
        for c in items:
            if cur and size + len(c["text"]) > BUNDLE_CHARS:
                bundles.append(cur)
                cur, size = [], 0
            cur.append(c)
            size += len(c["text"]) + 2
        if cur:
            bundles.append(cur)
    units = []
    for group in long_units + bundles:
        text = "\n\n".join(c["text"] for c in group)
        if len(text) < MIN_CHUNK_CHARS:
            continue
        uid = hashlib.sha1("|".join(c["chunk_id"] for c in group).encode()).hexdigest()[:16]
        n = max(1, min(MAX_QUESTIONS_PER_UNIT, round(len(text) / CHARS_PER_QUESTION)))
        units.append({"id": uid, "text": text, "n": n, "types": sorted({c["type"] for c in group}),
                      "source_url": group[0]["source_url"]})
    return units


def estimate(units):
    overhead = len(INSTRUCTIONS) // 4
    tin = sum(len(u["text"]) // 4 + overhead for u in units)
    tout = sum(u["n"] * 160 for u in units)  # ~160 tokens per Q&A pair as JSON
    credits = tin / 1e6 * GEN_PRICE_PER_M["input"] + tout / 1e6 * GEN_PRICE_PER_M["output"]
    return tin, tout, credits


def parse_pairs(text: str):
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return [d for d in data if isinstance(d, dict) and d.get("question") and d.get("answer")]


def generate(client, unit):
    path = CACHE / f"{unit['id']}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8")), (0, 0)
    resp = client.responses.create(
        model=GEN_MODEL,
        instructions=INSTRUCTIONS.replace("{n}", str(unit["n"])),
        input=f"SOURCE:\n{unit['text']}",
        temperature=0.7,
        max_output_tokens=2500,
    )
    pairs = parse_pairs(resp.output_text or "")
    usage = getattr(resp, "usage", None)
    tokens = (getattr(usage, "input_tokens", 0) or 0, getattr(usage, "output_tokens", 0) or 0)
    path.write_text(json.dumps({"unit": unit, "pairs": pairs}, ensure_ascii=False), encoding="utf-8")
    return {"unit": unit, "pairs": pairs}, tokens


def assemble():
    rows, seen = [], set()
    for path in sorted(CACHE.glob("*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        unit = rec["unit"]
        for p in rec["pairs"]:
            key = re.sub(r"\W+", " ", p["question"].lower()).strip()
            if key in seen:
                continue
            seen.add(key)
            cat = p.get("category") if p.get("category") in CATEGORIES else "other"
            url = p.get("source_url") or unit["source_url"]
            answer = p["answer"].strip()
            if "http" not in answer[-250:]:
                answer += f"\nSource: {url}"
            rows.append({"prompt": p["question"].strip(), "completion": answer,
                         "category": cat, "source_context": unit["text"][:6000],
                         "source_url": url,
                         "origin": "generated_from_source"})
    rng = random.Random(42)
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    train, held = [], []
    for items in by_cat.values():
        rng.shuffle(items)
        k = round(len(items) * HOLDOUT_SHARE)
        held += items[:k]
        train += items[k:]
    train += behavior_rows()
    rng.shuffle(train)

    # Real public questions (other people's posts) stay private: eval only, not in the repo.
    real_path = RAW / "qa_real.jsonl"
    real = [{"prompt": q["prompt"], "completion": q["completion"], "category": q["category"],
             "answer_status": q["answer_status"], "source_url": q["answer_source_urls"],
             "origin": "real_public_question"} for q in (read_jsonl(real_path) if real_path.exists() else [])]

    write_jsonl(SEED / "train_en.jsonl", train)
    write_jsonl(SEED / "eval_en.jsonl", held)
    write_jsonl(SEED / "eval_real_en.jsonl", real)
    print(f"train_en: {len(train)} rows ({len(train) - len(behavior_rows())} generated + {len(behavior_rows())} behavior)")
    print(f"eval_en: {len(held)} held-out rows; eval_real_en: {len(real)} real questions")
    print("categories:", dict(Counter(r["category"] for r in train).most_common()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="estimate only, no API calls")
    ap.add_argument("--limit", type=int, help="only process N randomly chosen units")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--assemble", action="store_true", help="only rebuild outputs from the cache")
    args = ap.parse_args()

    if args.assemble:
        return assemble()

    units = build_units()
    if args.limit:
        units = random.Random(7).sample(units, min(args.limit, len(units)))
    tin, tout, credits = estimate(units)
    print(f"{len(units)} source units -> ~{sum(u['n'] for u in units)} Q&A pairs")
    print(f"~{tin:,} input + ~{tout:,} output tokens on {GEN_MODEL} = ~{credits:.3f} credits")
    print("types:", dict(Counter(t for u in units for t in u["types"]).most_common()))
    if args.dry_run:
        return

    CACHE.mkdir(parents=True, exist_ok=True)
    client = get_client()
    done = used_in = used_out = failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(generate, client, u): u for u in units}
        for fut in as_completed(futures):
            try:
                _, (i, o) = fut.result()
                used_in += i
                used_out += o
            except Exception as e:  # keep going; failed units are retried on the next run
                failed += 1
                print(f"  unit {futures[fut]['id']} failed: {e}")
            done += 1
            if done % 25 == 0:
                print(f"  {done}/{len(units)} units")
    spent = used_in / 1e6 * GEN_PRICE_PER_M["input"] + used_out / 1e6 * GEN_PRICE_PER_M["output"]
    print(f"done, {failed} failed. Reported usage: {used_in:,} in / {used_out:,} out tokens = ~{spent:.3f} credits")
    assemble()


if __name__ == "__main__":
    main()
