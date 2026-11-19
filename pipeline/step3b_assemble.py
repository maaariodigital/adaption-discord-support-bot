"""Step 3b: clean and merge every Adaptive Data output into one training set,
then upload it for AutoScientist.

Cleanup:
- Links: docs '/index.md' paths become page URLs; the made-up docs.adaption.ai
  becomes docs.adaptionlabs.ai.
- Hand-written behavior rows: adapting rewrote one into a false claim (SLA), so
  the adapted copies are dropped and all hand-written answers are used as written.
- Answers that talk about "the provided context" (which users never see) are
  rewritten by Adaption's hosted model to refer to Adaption's public docs.
Then rows are deduplicated, written to data/final/train.jsonl and, with
--upload, uploaded in raw mode (free) so AutoScientist trains on prompt/completion.

    python pipeline/step3b_assemble.py --dry-run    # counts + cost quote, no API calls
    python pipeline/step3b_assemble.py --limit 10   # rewrite 10 rows, show before/after
    python pipeline/step3b_assemble.py              # full cleanup (quoted, confirm)
    python pipeline/step3b_assemble.py --upload     # cleanup + upload for training
"""
import argparse
import hashlib
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from behavior_rows import behavior_rows
from common import confirm_spend, get_client, load_state, read_jsonl, save_state, write_jsonl
from config import ADAPTED, GEN_MODEL, GEN_PRICE_PER_M, ROOT, WORK

FINAL = ROOT / "data" / "final"
CACHE = WORK / "clean_cache"

# Phrases about the hidden reference material, per language. A match only sends
# the row to the rewrite pass; the model leaves it unchanged if nothing applies.
CONTEXT_MENTION = re.compile(
    r"provided (context|information|documentation|docs|text|sources?|material)|"
    r"the context (does|doesn)|given context|in the context|"
    r"प्रदान (की गई|किए गए)|दिए गए संदर्भ|संदर्भ में|"
    r"فراہم کردہ|دیے گئے سیاق|سیاق و سباق|"
    r"ibinigay na|sa konteksto|"
    r"muktadha|uliotolewa|iliyotolewa|zilizotolewa",
    re.I,
)

REWRITE_INSTRUCTIONS = (
    "You edit replies written by the support bot of the Adaption Labs Discord. Users never "
    "see the bot's reference material, so phrases such as 'the provided context', 'the "
    "provided documentation' or 'the given information' (in any language) make no sense to "
    "them. Rewrite only those phrases so they refer to Adaption's public docs or say the "
    "information isn't public. Example: 'The provided context does not say X' becomes "
    "'Adaption's public docs don't cover X'. Keep everything else exactly as written: same "
    "language, facts, links, code, formatting and length. If there is nothing to change, "
    "return the reply unchanged. Output only the reply."
)


def fix_links(text: str) -> str:
    text = re.sub(r"https?://docs\.adaption\.ai\b", "https://docs.adaptionlabs.ai", text)
    return re.sub(r"(https://docs\.adaptionlabs\.ai[^\s)\]>\"'`]*?)/index\.md", r"\1/", text)


def load_sources():
    """Return normalized rows plus a per-source count."""
    rows, counts = [], {}
    for name in ["adapted_multilingual.jsonl", "adapted_multilingual_part2.jsonl", "programs_multilingual.jsonl"]:
        path = ADAPTED / name
        if not path.exists():
            continue
        n = 0
        for r in read_jsonl(path):
            if r.get("origin") == "behavior_handwritten":
                continue  # replaced by the hand-written originals below
            rows.append({"prompt": r.get("prompt"), "completion": r.get("enhanced_completion"),
                         "language": r.get("target_language_code") or "original",
                         "origin": "programs" if name.startswith("programs") else "seed",
                         "category": r.get("category", "")})
            n += 1
        counts[name] = n
    for path in sorted(ADAPTED.glob("invented_*.jsonl")):
        key = path.stem.removeprefix("invented_")
        n = 0
        for r in read_jsonl(path):
            rows.append({"prompt": r.get("enhanced_prompt"), "completion": r.get("enhanced_completion"),
                         "language": key, "origin": "invented", "category": ""})
            n += 1
        counts[path.name] = n
    for b in behavior_rows():
        rows.append({"prompt": b["prompt"], "completion": b["completion"], "language": "original",
                     "origin": "behavior_handwritten", "category": b["category"]})
    counts["behavior_rows.py"] = len(behavior_rows())
    return rows, counts


def cache_path(text: str):
    return CACHE / f"{hashlib.sha1(text.encode('utf-8')).hexdigest()}.json"


def rewrite(client, text: str):
    path = cache_path(text)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["out"], (0, 0)
    resp = client.responses.create(model=GEN_MODEL, instructions=REWRITE_INSTRUCTIONS,
                                   input=text, temperature=0.2, max_output_tokens=2000)
    out = (resp.output_text or "").strip()
    usage = getattr(resp, "usage", None)
    tokens = (getattr(usage, "input_tokens", 0) or 0, getattr(usage, "output_tokens", 0) or 0)
    # Guard against truncated or runaway output: keep the original if length drifts a lot.
    if not out or not (0.6 <= len(out) / max(1, len(text)) <= 1.4):
        out = text
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"in": text, "out": out}, ensure_ascii=False), encoding="utf-8")
    return out, tokens


def quote_credits(texts):
    # Non-Latin scripts take more tokens per character; x2 on output covers reasoning tokens.
    tok = sum(len(t) / (2.0 if re.search(r"[ऀ-ॿ؀-ۿ]", t) else 3.5) for t in texts)
    tin = tok + 180 * len(texts)
    tout = tok * 2
    return (tin * GEN_PRICE_PER_M["input"] + tout * GEN_PRICE_PER_M["output"]) / 1e6


def upload(client, state, path):
    sha = hashlib.sha1(path.read_bytes()).hexdigest()
    if state.get("train_dataset_id") and state.get("train_file_sha") == sha:
        print(f"Training set unchanged; reusing {state['train_dataset_id']}")
        return
    if state.get("train_dataset_id"):
        state.setdefault("unused_dataset_ids", []).append(state["train_dataset_id"])
    ds = client.datasets.upload_file(str(path), name="adaption-support-train-final", processing_mode="raw",
                                     column_mapping={"prompt": "prompt", "completion": "completion"})
    st = client.datasets.wait_for_completion(ds.dataset_id, timeout=3600)
    if st.status == "failed":
        raise SystemExit(f"Upload failed: {st.error_data}")
    state["train_dataset_id"] = ds.dataset_id
    state["train_file_sha"] = sha
    save_state(state)
    print(f"Uploaded training set -> {ds.dataset_id} ({st.row_count} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, help="rewrite only this many rows and show before/after")
    ap.add_argument("--upload", action="store_true", help="upload the final set for AutoScientist")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    rows, counts = load_sources()
    print("Sources:", json.dumps(counts))
    if "adapted_multilingual_part2.jsonl" not in counts:
        print("Note: part 2 of the seed isn't downloaded yet (step2_adapt.py --download --part2).")

    for r in rows:
        r["completion"] = fix_links(r["completion"] or "")
    rows = [r for r in rows if r["prompt"] and r["completion"]]

    todo = [r for r in rows if r["origin"] != "behavior_handwritten" and CONTEXT_MENTION.search(r["completion"])]
    pending = [r for r in todo if not cache_path(r["completion"]).exists()]
    if args.limit:
        random.Random(1).shuffle(pending)
        pending = pending[: args.limit]
    est = quote_credits([r["completion"] for r in pending])
    print(f"Rows mentioning the hidden context: {len(todo)} ({len(todo) - len(pending) if not args.limit else 'n/a'} cached)")
    if args.dry_run:
        print(f"Rewrite pass would cost ~{est:.3f} credits with {GEN_MODEL}. Dry run, stopping.")
        return

    if pending:
        quote = f"{len(pending)} rows with {GEN_MODEL}\nEstimated credits: ~{est:.3f} (hosted model, billed per token)"
        if not confirm_spend("Responses API: rewrite 'provided context' phrases", quote, args.yes or est < 0.05):
            return
        client = get_client()
        used = [0, 0]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(rewrite, client, r["completion"]): r for r in pending}
            for i, f in enumerate(as_completed(futs), 1):
                try:
                    _, (a, b) = f.result()
                    used[0] += a
                    used[1] += b
                except Exception as e:  # leave the row as-is; a rerun retries it
                    print(f"  rewrite failed: {e}")
                if i % 50 == 0:
                    print(f"  {i}/{len(pending)}")
        spent = (used[0] * GEN_PRICE_PER_M["input"] + used[1] * GEN_PRICE_PER_M["output"]) / 1e6
        print(f"Rewrite usage: {used[0]:,} in / {used[1]:,} out tokens = ~{spent:.3f} credits")

    if args.limit:
        for r in pending:
            p = cache_path(r["completion"])
            if p.exists():
                rec = json.loads(p.read_text(encoding="utf-8"))
                print(f"\n[{r['language']}] BEFORE: {rec['in'][:300]}\n        AFTER:  {rec['out'][:300]}")
        return

    for r in todo:
        p = cache_path(r["completion"])
        if p.exists():
            r["completion"] = json.loads(p.read_text(encoding="utf-8"))["out"]

    seen, final = set(), []
    for r in rows:
        key = (r["prompt"].strip().lower(), r["language"])
        if key not in seen:
            seen.add(key)
            final.append(r)
    random.Random(13).shuffle(final)
    out = FINAL / "train.jsonl"
    write_jsonl(out, final)
    by_lang = {}
    for r in final:
        by_lang[r["language"]] = by_lang.get(r["language"], 0) + 1
    left = sum(bool(CONTEXT_MENTION.search(r["completion"])) for r in final if r["origin"] != "behavior_handwritten")
    print(f"Wrote {out}: {len(final)} rows {json.dumps(by_lang)}; {left} still match the context pattern")

    if args.upload:
        upload(get_client(), load_state(), out)


if __name__ == "__main__":
    main()
