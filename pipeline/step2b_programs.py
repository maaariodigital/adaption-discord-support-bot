"""Step 2b: grow the thin community-program topics with Adaptive Data.

The seed has few rows on ambassadors, grants, startups, Discord and pricing.
This step uploads a balanced subset of those rows, uses Expand -> similar rows
(datasets.augment) to generate more in the same style, keeps the on-topic
new questions (Expand's own answers drift and invent facts), attaches the matching
knowledge-base section as context, then adapts and localizes with web-search
grounding (hallucination mitigation) on, so Adaptive Data writes the answers.

    python pipeline/step2b_programs.py               # quote + confirm each paid stage
    python pipeline/step2b_programs.py --download    # save results locally
"""
import argparse
import json
import random
import re
import time
from collections import defaultdict

from common import confirm_spend, get_client, load_state, read_jsonl, save_state, write_jsonl
from config import (ADAPTED, BLUEPRINT, LOCALIZE_PAIRS, LOCALIZE_SAMPLE_RATE,
                    PROGRAM_CATEGORIES, PROGRAM_CAP_PER_CATEGORY, PROGRAM_SIMILAR_ROWS,
                    KNOWLEDGE, SEED, WORK)


def build_subset():
    rows = [r for r in read_jsonl(SEED / "train_en.jsonl") if r["category"] in PROGRAM_CATEGORIES]
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    rng = random.Random(11)
    subset = []
    for items in by_cat.values():  # cap big categories so similar rows follow a balanced mix
        rng.shuffle(items)
        subset += items[:PROGRAM_CAP_PER_CATEGORY]
    path = WORK / "programs_subset_en.jsonl"
    write_jsonl(path, [{"prompt": r["prompt"], "completion": r["completion"], "category": r["category"]}
                       for r in subset])
    print(f"Programs subset: {len(subset)} rows {dict((k, min(len(v), PROGRAM_CAP_PER_CATEGORY)) for k, v in by_cat.items())}")
    return path


# Expand questions are kept only if they touch a program topic. The first matching
# topic picks the knowledge-base sections passed as context.
TOPICS = [
    (r"ambassador", ["6. Community programs", "5. Support"]),
    (r"grant", ["6. Community programs", "5. Support"]),
    (r"hackathon|hack |challenge|competition|judg|prize|submission", ["6. Community programs", "5. Support"]),
    (r"startup|start-up", ["6. Community programs", "5. Support"]),
    (r"discord|community", ["5. Support", "6. Community programs"]),
    (r"credit|pricing|price|billing|free tier|subscription|refund", ["4. Pricing", "5. Support"]),
]


def kb_sections():
    parts = re.split(r"(?m)^## ", (KNOWLEDGE / "knowledge_base.md").read_text(encoding="utf-8"))
    return {p.split("\n", 1)[0].strip(): "## " + p.strip() for p in parts[1:]}


def curate(client, state):
    raw = WORK / "programs_expanded.jsonl"
    if not raw.exists():
        client.datasets.download(state["programs_augmented_id"], file_format="jsonl").write_to_file(str(raw))
    kb = kb_sections()
    seen = {r["prompt"].strip().lower() for r in read_jsonl(SEED / "train_en.jsonl")}
    out = []
    for r in read_jsonl(raw):
        prompt = (r.get("original_prompt") or "").strip()
        if not prompt or prompt.lower() in seen:
            continue  # drops the source rows, which the main adapt run already covers
        for pattern, names in TOPICS:
            if re.search(pattern, prompt.lower()):
                ctx = "\n\n".join(v for n in names for k, v in kb.items() if k.startswith(n))
                out.append({"prompt": prompt, "source_context": ctx})
                seen.add(prompt.lower())
                break
    path = WORK / "programs_curated_en.jsonl"
    write_jsonl(path, out)
    print(f"Curated {len(out)} on-topic questions from Expand (answers dropped, KB context attached)")
    return path


def wait_ready(client, dsid, label):
    st = client.datasets.wait_for_completion(dsid, timeout=4 * 3600)
    if st.status == "failed":
        raise SystemExit(f"{label} failed: {st.error_data}")
    print(f"{label}: {st.status}, {st.row_count} rows")
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--only-expand", action="store_true", help="stop after the similar-rows stage")
    args = ap.parse_args()
    client = get_client()
    state = load_state()

    if args.download:
        dsid = state["programs_adapted_id"]
        wait_ready(client, dsid, "Programs adapt")
        out = ADAPTED / "programs_multilingual.jsonl"
        ADAPTED.mkdir(parents=True, exist_ok=True)
        client.datasets.download(dsid, file_format="jsonl").write_to_file(str(out))
        ev = client.datasets.get_evaluation(dsid)
        (ADAPTED / "programs_evaluation.json").write_text(
            json.dumps(ev.model_dump(), indent=2, default=str), encoding="utf-8")
        print(f"Saved {out}; evaluation {ev.status}: {ev.quality}")
        return

    # 1. Upload the balanced subset (free). Raw mode sets the columns, which Expand needs.
    if not state.get("programs_dataset_id"):
        ds = client.datasets.upload_file(str(build_subset()), name="adaption-support-programs-en",
                                         processing_mode="raw",
                                         column_mapping={"prompt": "prompt", "completion": "completion"})
        state["programs_dataset_id"] = ds.dataset_id
        save_state(state)
        wait_ready(client, ds.dataset_id, "Programs import")

    # 2. Expand: generate similar rows.
    if not state.get("programs_augmented_id"):
        src = state["programs_dataset_id"]
        est = client.datasets.augment(src, similar_rows=PROGRAM_SIMILAR_ROWS, estimate=True)
        quote = f"Up to {PROGRAM_SIMILAR_ROWS} similar rows\nEstimated credits: {est.estimated_credits_consumed}"
        if not confirm_spend("Expand: similar rows for community programs", quote, args.yes):
            return
        resp = client.datasets.augment(src, similar_rows=PROGRAM_SIMILAR_ROWS,
                                       idempotency_key=f"programs-similar-{src}-{PROGRAM_SIMILAR_ROWS}")
        state["programs_augmented_id"] = resp.dataset_id
        save_state(state)
        wait_ready(client, resp.dataset_id, "Similar rows")

    if args.only_expand:
        return

    # 3. Curate (free): Expand writes useful new questions, but its answers drift off
    #    Adaption and invent facts and links. Keep on-topic questions only, drop the
    #    answers and attach the matching knowledge-base section as context.
    if not state.get("programs_curated_id"):
        ds = client.datasets.upload_file(str(curate(client, state)), name="adaption-support-programs-curated")
        state["programs_curated_id"] = ds.dataset_id
        save_state(state)
        while (st := client.datasets.get_status(ds.dataset_id)).row_count is None:
            if st.status == "failed":
                raise SystemExit(f"Curated import failed: {st.error_data}")
            time.sleep(5)
        print(f"Curated import: {st.row_count} rows")

    # 4. Adapt + localize: Adaptive Data writes the answers from the context, grounded with web search.
    if not state.get("programs_adapted_id"):
        dsid = state["programs_curated_id"]
        kw = {
            "column_mapping": {"prompt": "prompt", "context": ["source_context"]},
            "recipe_specification": {"recipes": {"deduplication": True, "prompt_rephrase": False}},
            "brand_controls": {"length": "concise", "blueprint": BLUEPRINT, "hallucination_mitigation": True},
            "localize": {"pairs": LOCALIZE_PAIRS, "sample_rate": LOCALIZE_SAMPLE_RATE},
            # Explicit, since an omitted max_rows can reuse a previous run's limit.
            "job_specification": {"max_rows": client.datasets.get_status(dsid).row_count},
        }
        est = client.datasets.adapt(dsid, estimate=True, **kw)
        quote = (f"Adapt + localize the expanded programs set, hallucination mitigation on\n"
                 f"Estimated credits: {est.estimated_credits_consumed}\nEstimated time: {est.estimated_minutes} min")
        if not confirm_spend("Adaptive Data: programs adapt + localize", quote, args.yes):
            return
        run = client.datasets.adapt(dsid, **kw)
        state["programs_adapted_id"] = dsid
        state["programs_adapt_run_id"] = run.run_id
        save_state(state)
        print(f"Run started: {run.run_id}. Fetch results later with --download.")


if __name__ == "__main__":
    main()
