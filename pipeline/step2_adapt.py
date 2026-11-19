"""Step 2: upload the English seed to Adaptive Data, adapt it and localize it
into Hindi (IN), Urdu (PK), Tagalog (PH) and Swahili (KE).

    python pipeline/step2_adapt.py --max-rows 200   # quote, then a small test run
    python pipeline/step2_adapt.py                  # quote, then the full run
    python pipeline/step2_adapt.py --download       # fetch results + evaluation of the last run

Always asks for a quote first and waits for you to type 'spend'.
"""
import argparse
import json
import time

from common import confirm_spend, get_client, load_state, read_jsonl, save_state, write_jsonl
from config import (ADAPTED, BLUEPRINT, HALLUCINATION_MITIGATION, LOCALIZE_PAIRS,
                    LOCALIZE_SAMPLE_RATE, SEED)


def adapt_kwargs(max_rows):
    kw = {
        "column_mapping": {"prompt": "prompt", "completion": "completion", "context": ["source_context"]},
        # No prompt_rephrase: it turned user questions into "Analyze the provided
        # documentation..." tasks. The model should learn from real Discord wording.
        "recipe_specification": {"recipes": {"deduplication": True, "prompt_rephrase": False}},
        "brand_controls": {"length": "concise", "blueprint": BLUEPRINT,
                           "hallucination_mitigation": HALLUCINATION_MITIGATION},
        "localize": {"pairs": LOCALIZE_PAIRS, "sample_rate": LOCALIZE_SAMPLE_RATE},
    }
    if max_rows:
        kw["job_specification"] = {"max_rows": max_rows}
    return kw


# The free plan caps each launch at about 1,000 rows, and the quote doesn't show it:
# a 1,894-row launch was charged in full but processed 999 rows (2026-11-17).
# Part 2 uploads the seed rows the first run didn't reach.
PARTS = {
    1: {"key": "seed_dataset_id", "file": "train_en.jsonl", "name": "adaption-support-seed-en", "suffix": ""},
    2: {"key": "seed2_dataset_id", "file": "train_en_part2.jsonl", "name": "adaption-support-seed-en-part2", "suffix": "_part2"},
}
PLAN_ROW_LIMIT = 1000


def build_part2():
    done = {r["prompt"] for r in read_jsonl(ADAPTED / "adapted_multilingual.jsonl")
            if r.get("target_language_code") == "original"}
    rest = [r for r in read_jsonl(SEED / "train_en.jsonl") if r["prompt"] not in done]
    write_jsonl(SEED / PARTS[2]["file"], rest)
    print(f"Part 2: {len(rest)} seed rows not covered by the first run")


def upload(client, state, part):
    cfg = PARTS[part]
    if state.get(cfg["key"]):
        print(f"Reusing uploaded dataset {state[cfg['key']]}")
        return state[cfg["key"]]
    if part == 2:
        build_part2()
    path = SEED / cfg["file"]
    if not path.exists():
        raise SystemExit(f"{path} not found. Run step1_build_seed.py first.")
    ds = client.datasets.upload_file(str(path), name=cfg["name"])
    print(f"Uploaded {path.name} -> {ds.dataset_id}; waiting for import...")
    while True:
        st = client.datasets.get_status(ds.dataset_id)
        if st.status == "failed":
            raise SystemExit(f"Import failed: {st.error_data}")
        if st.row_count is not None:
            print(f"Imported {st.row_count} rows")
            break
        time.sleep(5)
    state[cfg["key"]] = ds.dataset_id
    save_state(state)
    return ds.dataset_id


def download(client, state, part):
    cfg = PARTS[part]
    dsid = state.get(cfg["key"])
    st = client.datasets.wait_for_completion(dsid, timeout=4 * 3600)
    print(f"Adaptive Data status: {st.status}")
    ADAPTED.mkdir(parents=True, exist_ok=True)
    out = ADAPTED / f"adapted_multilingual{cfg['suffix']}.jsonl"
    client.datasets.download(dsid, file_format="jsonl").write_to_file(str(out))
    print(f"Saved {out}")
    ev = client.datasets.get_evaluation(dsid)
    (ADAPTED / f"evaluation{cfg['suffix']}.json").write_text(
        json.dumps(ev.model_dump(), indent=2, default=str), encoding="utf-8")
    print(f"Evaluation ({ev.status}): {ev.quality}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rows", type=int, help="adapt only a random sample (test run)")
    ap.add_argument("--part2", action="store_true", help="adapt the seed rows the first run didn't reach")
    ap.add_argument("--yes", action="store_true", help="skip the interactive spend prompt")
    ap.add_argument("--download", action="store_true", help="wait for the run and download results")
    args = ap.parse_args()
    part = 2 if args.part2 else 1

    client = get_client()
    state = load_state()
    if args.download:
        return download(client, state, part)

    dsid = upload(client, state, part)
    # Always send max_rows: if it's omitted, the server reuses the previous run's
    # limit (seen on 2026-11-17: a "full" quote came back at the old 50-row price).
    if not args.max_rows:
        args.max_rows = client.datasets.get_status(dsid).row_count
    if args.max_rows > PLAN_ROW_LIMIT:
        raise SystemExit(f"{args.max_rows} rows is over the ~{PLAN_ROW_LIMIT}-row per-launch limit; "
                         "the extra rows would be charged but not processed. Split the dataset.")
    kw = adapt_kwargs(args.max_rows)
    est = client.datasets.adapt(dsid, estimate=True, **kw)
    locales = ", ".join(f"{p['language']}-{p['country']}" for p in LOCALIZE_PAIRS)
    quote = (f"Part {part}: {args.max_rows} English rows + localized copies ({locales}, sample rate {LOCALIZE_SAMPLE_RATE})\n"
             f"Estimated credits: {est.estimated_credits_consumed}\n"
             f"Estimated time: {est.estimated_minutes} min")
    if not confirm_spend("Adaptive Data: adapt + localize", quote, args.yes):
        return
    run = client.datasets.adapt(dsid, **kw)
    state[f"adapt{cfg_suffix(part)}_run_id"] = run.run_id
    state[f"adapt{cfg_suffix(part)}_max_rows"] = args.max_rows
    save_state(state)
    print(f"Run started: {run.run_id}. Fetch results later with --download{' --part2' if part == 2 else ''}.")


def cfg_suffix(part):
    return PARTS[part]["suffix"]


if __name__ == "__main__":
    main()
