"""Step 3c: a "docs in the prompt" training set for AutoScientist run 4.

Runs 1-3 trained on bare question -> answer pairs, so the model had to memorize
Adaption's docs. The live bot doesn't work that way: it retrieves docs first and
answers from them. This step rebuilds data/final/train.jsonl so each prompt
carries the docs the bot would see, and the model learns to answer from them.

For each row:
- Context: the chunks behind the answer's source link (when the link is in
  knowledge/), topped up with keyword search, up to K chunks, in random order.
- Rows whose links all point outside knowledge/ (news, third-party sites) are
  dropped: their facts aren't in any context, so they'd teach the model to
  state things the docs don't say.
- Code-switched (Invent) rows are re-answered from that context with the bot's
  own instructions. Most of them handed off ("not in my context") although the
  docs cover the question, and many linked made-up domains (adaption.io).

    python pipeline/step3c_grounded.py --dry-run      # counts, sizes, cost quote; no API calls
    python pipeline/step3c_grounded.py --limit 5      # re-answer 5 rows, show before/after
    python pipeline/step3c_grounded.py                # full build (quoted, confirm)
    python pipeline/step3c_grounded.py --upload       # build + upload for step4 --dataset grounded
"""
import argparse
import hashlib
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from common import confirm_spend, get_client, load_state, read_jsonl, save_state, write_jsonl
from config import BLUEPRINT, GEN_MODEL, GEN_PRICE_PER_M, ROOT, WORK

sys.path.insert(0, str(ROOT / "bot"))
from answer import STYLE_HINT, clean, script_hint  # noqa: E402  same instructions as the live bot
from knowledge import Knowledge  # noqa: E402

FINAL = ROOT / "data" / "final"
CACHE = WORK / "grounded_cache"
K = 4             # chunks per prompt (the bot uses 6; fewer keeps training tokens down)
CHUNK_CHARS = 1000
URL = re.compile(r"https?://[^\s)\]>\"'`*]+")
OWN_DOMAINS = re.compile(r"https?://([a-z0-9-]+\.)*(adaptionlabs\.ai|hackyard\.)", re.I)
# Prompt layout the fine-tuned model learns; the bot must send the same layout to use it.
PROMPT_TEMPLATE = "Adaption's public docs (link the most relevant URL):\n{context}\n\nQuestion: {question}"


def norm(u: str) -> str:
    return u.rstrip("/.,)")


def context_for(kb: Knowledge, by_url: dict, row: dict, rng: random.Random) -> list:
    gold = []
    for u in URL.findall(row["completion"]):
        for c in by_url.get(norm(u), []):
            if c not in gold:
                gold.append(c)
    if len(gold) > 2:  # a long page: keep the parts that match the question best
        q = kb.search(row["prompt"] + " " + row["completion"], k=40)
        gold = sorted(gold, key=lambda c: q.index(c) if c in q else 99)[:2]
    # Latin words in the answer (product names, English terms) help search for non-English rows.
    query = row["prompt"] + " " + " ".join(re.findall(r"[A-Za-z][A-Za-z0-9_.-]{2,}", row["completion"]))
    chunks = gold + [c for c in kb.search(query, k=K + 2) if c not in gold]
    chunks = chunks[:K]
    rng.shuffle(chunks)  # the answer's chunk shouldn't always come first
    return chunks


def trim(chunks):
    """Like the bot's format_context, with shorter chunks."""
    return "\n\n".join(f"[{i}] {c.title} | {c.url}\n{c.text[:CHUNK_CHARS]}" for i, c in enumerate(chunks, 1))


def cache_path(row):
    return CACHE / f"{hashlib.sha1((row['prompt'] + row['language']).encode('utf-8')).hexdigest()}.json"


def reanswer(client, row, context):
    path = cache_path(row)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["out"], 0.0
    instructions = (BLUEPRINT + STYLE_HINT + "\n\nThe user writes in a code-switched mix. Reply in the same mix "
                    "and register as the user, not in pure English or pure local language. End with the most "
                    "relevant URL from the context." + "\n\nCONTEXT (Adaption's public docs, blog and site):\n"
                    + context + script_hint(row["prompt"]))
    r = client.responses.create(model=GEN_MODEL, instructions=instructions, input=row["prompt"],
                                temperature=0.3, max_output_tokens=2000)
    out = (r.output_text or "").strip()
    if not out:
        raise RuntimeError(f"empty answer ({r.status})")
    out = clean(out)
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"in": row["completion"], "out": out}, ensure_ascii=False), encoding="utf-8")
    return out, getattr(r.usage, "cost", 0) or 0


def upload(client, state, path):
    sha = hashlib.sha1(path.read_bytes()).hexdigest()
    if state.get("train_dataset_id_grounded") and state.get("train_file_sha_grounded") == sha:
        print(f"Grounded set unchanged; reusing {state['train_dataset_id_grounded']}")
        return
    ds = client.datasets.upload_file(str(path), name="adaption-support-train-grounded", processing_mode="raw",
                                     column_mapping={"prompt": "prompt", "completion": "completion"})
    st = client.datasets.wait_for_completion(ds.dataset_id, timeout=3600)
    if st.status == "failed":
        raise SystemExit(f"Upload failed: {st.error_data}")
    state["train_dataset_id_grounded"] = ds.dataset_id
    state["train_file_sha_grounded"] = sha
    save_state(state)
    print(f"Uploaded grounded training set -> {ds.dataset_id} ({st.row_count} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, help="re-answer only this many code-switched rows and show them")
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()

    rows = read_jsonl(FINAL / "train.jsonl")
    kb = Knowledge()
    by_url = {}
    for c in kb.chunks:
        if c.url:
            by_url.setdefault(norm(c.url), []).append(c)

    kept, dropped = [], 0
    for r in rows:
        urls = [norm(u) for u in URL.findall(r["completion"])]
        grounded = (not urls or any(u in by_url for u in urls) or any(OWN_DOMAINS.match(u) for u in urls)
                    or r["origin"] in ("invented", "behavior_handwritten"))  # Invent rows are re-answered
        if grounded:
            kept.append(r)
        else:
            dropped += 1
    rng = random.Random(4)
    for r in kept:
        r["_ctx"] = trim(context_for(kb, by_url, r, rng))

    todo = [r for r in kept if r["origin"] == "invented"]
    pending = [r for r in todo if not cache_path(r).exists()]
    if args.limit:
        random.Random(1).shuffle(pending)
        pending = pending[: args.limit]
    est_in = sum(len(BLUEPRINT) + len(STYLE_HINT) + len(r["_ctx"]) + len(r["prompt"]) for r in pending) / 3.5
    est = (est_in * GEN_PRICE_PER_M["input"] + len(pending) * 1200 * GEN_PRICE_PER_M["output"]) / 1e6
    prompt_chars = [len(PROMPT_TEMPLATE.format(context=r["_ctx"], question=r["prompt"])) for r in kept]
    print(f"Rows: {len(rows)} in, {dropped} dropped (links only outside knowledge/), {len(kept)} kept")
    print(f"Code-switched rows to re-answer: {len(todo)} ({len(todo) - len(pending) if not args.limit else 'n/a'} cached)")
    print(f"Prompt size: median {sorted(prompt_chars)[len(prompt_chars) // 2]:,} chars; "
          f"~{sum(prompt_chars) / 3.5 / 1e6:.1f}M prompt tokens per epoch")
    if args.dry_run:
        print(f"Re-answering would cost ~{est:.3f} credits with {GEN_MODEL}. Dry run, stopping.")
        return

    if pending:
        quote = f"{len(pending)} code-switched rows with {GEN_MODEL}\nEstimated credits: ~{est:.3f}"
        if not confirm_spend("Responses API: re-answer code-switched rows from the docs", quote, args.yes):
            return
        client = get_client().with_options(max_retries=6, timeout=120)
        spent = 0.0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(reanswer, client, r, r["_ctx"]): r for r in pending}
            for i, f in enumerate(as_completed(futs), 1):
                try:
                    spent += f.result()[1]
                except Exception as e:  # keep the old answer out; a rerun retries it
                    print(f"  re-answer failed: {e}")
                if i % 50 == 0:
                    print(f"  {i}/{len(pending)}")
        print(f"Re-answer cost: {spent:.4f} credits (as reported by the API)")

    if args.limit:
        for r in pending:
            p = cache_path(r)
            if p.exists():
                rec = json.loads(p.read_text(encoding="utf-8"))
                print(f"\n[{r['language']}] Q: {r['prompt'][:200]}\n  BEFORE: {rec['in'][:300]}\n  AFTER:  {rec['out'][:400]}")
        return

    final, missing = [], 0
    for r in kept:
        if r["origin"] == "invented":
            p = cache_path(r)
            if not p.exists():
                missing += 1
                continue
            r["completion"] = json.loads(p.read_text(encoding="utf-8"))["out"]
        final.append({"prompt": PROMPT_TEMPLATE.format(context=r["_ctx"], question=r["prompt"]),
                      "completion": r["completion"], "language": r["language"], "origin": r["origin"],
                      "category": r["category"]})
    out = FINAL / "train_grounded.jsonl"
    write_jsonl(out, final)
    print(f"Wrote {out}: {len(final)} rows" + (f" ({missing} code-switched rows left out: re-answer failed)" if missing else ""))
    if args.upload:
        upload(get_client(), load_state(), out)


if __name__ == "__main__":
    main()
