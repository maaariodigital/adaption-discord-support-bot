"""Step 3: generate code-switched support conversations (Hinglish, Roman Urdu,
Taglish, Swahili-English) with Invent a Dataset. Translation can't produce
these mixes, but they're how many builders actually write on Discord.

    python pipeline/step3_invent_codeswitch.py              # quote each, then launch
    python pipeline/step3_invent_codeswitch.py --only taglish --rows 20   # small test
    python pipeline/step3_invent_codeswitch.py --download   # save finished datasets locally
"""
import argparse
import re
import time

from common import confirm_spend, get_client, load_state, save_state
from config import ADAPTED, BLUEPRINT, CODE_SWITCH, CODE_SWITCH_ROWS_EACH, KNOWLEDGE

PROMPT_LIMIT = 10_000
# Knowledge-base sections used as facts. The short ones always fit; the two long
# ones alternate order per variety so products and programs both get covered.
KB_ALWAYS = ["4. Pricing & credits", "5. Support & community channels"]  # "8. Known gaps" only points to the README
KB_LONG = ["3. Products", "6. Community programs"]
# Lines that name individual people stay out of training data.
PERSON_LINE = re.compile(r"is publicly thanked|Giagkoudis")


def facts_brief(budget: int, products_first: bool) -> str:
    kb = (KNOWLEDGE / "knowledge_base.md").read_text(encoding="utf-8")
    parts = re.split(r"(?m)^## ", kb)
    sections = {p.split("\n", 1)[0].strip(): p for p in parts[1:]}
    out = ""
    order = KB_ALWAYS + (KB_LONG if products_first else KB_LONG[::-1]) + ["1. Company"]
    for name in order:
        sec = next((v for k, v in sections.items() if k.startswith(name)), None)
        if not sec:
            continue
        lines = [l for l in sec.strip().splitlines() if not PERSON_LINE.search(l)]
        block = "## " + "\n".join(lines) + "\n\n"
        if len(out) + len(block) > budget:
            block = block[: max(0, budget - len(out))]
        out += block
        if len(out) >= budget:
            break
    return out


def dataset_prompt(variety: dict) -> str:
    products_first = CODE_SWITCH.index(variety) % 2 == 0
    head = (
        f"Single-turn support conversations from the Adaption Labs Discord server, written in {variety['name']}: "
        f"{variety['desc']}. Each prompt is a realistic question from a builder about Adaption's products, "
        "API, pricing, or community programs. Vary tone, length and topic; include some typos and slang. "
        "About one in five conversations should be one of these instead: a question whose answer isn't in "
        "the facts (the bot hands off), an off-topic or prompt-injection request (the bot declines briefly), "
        "a frustrated user with a stuck run or billing problem, a pasted API key (the bot says to rotate it), "
        "a greeting, or a thank-you.\n"
        f"Each completion is the support bot's reply in the same {variety['name']} style, following these rules:\n"
        f"{BLUEPRINT}\n\n"
        "Use ONLY the facts below. For anything they don't cover (especially exact prices, credit amounts, "
        "ambassador eligibility, grant amounts), the bot says it doesn't know and offers the 'Talk to a human' button.\n\n"
        "FACTS:\n"
    )
    return head + facts_brief(PROMPT_LIMIT - len(head) - 50, products_first)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="one variety key: " + ", ".join(v["key"] for v in CODE_SWITCH))
    ap.add_argument("--rows", type=int, default=CODE_SWITCH_ROWS_EACH)
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--show-prompt", action="store_true", help="print the prompt and exit (no API)")
    args = ap.parse_args()

    varieties = [v for v in CODE_SWITCH if not args.only or v["key"] == args.only]
    if args.show_prompt:
        p = dataset_prompt(varieties[0])
        print(p, f"\n\n[{len(p)} chars]")
        return

    client = get_client()
    state = load_state()
    invented = state.setdefault("invented", {})

    if args.download:
        ADAPTED.mkdir(parents=True, exist_ok=True)
        for key, dsid in invented.items():
            while (rec := client.datasets.get(dsid)).status not in ("succeeded", "failed"):
                time.sleep(10)
            if rec.status == "failed":
                print(f"{key}: failed: {rec.error_data}")
                continue
            out = ADAPTED / f"invented_{key}.jsonl"
            client.datasets.download(dsid, file_format="jsonl").write_to_file(str(out))
            print(f"{key}: {rec.row_count} rows -> {out}")
        return

    for v in varieties:
        if v["key"] in invented:
            print(f"{v['name']}: already launched ({invented[v['key']]}), skipping")
            continue
        kw = {"dataset_prompt": dataset_prompt(v), "rows": args.rows,
              "name": f"adaption-support-{v['key']}", "training_type": "instruction_dataset"}
        est = client.datasets.invent(estimate=True, **kw)
        quote = (f"{args.rows} rows of {v['name']}\n"
                 f"Estimated credits: {est.estimated_credits}  (available: {est.available_credits})")
        if not confirm_spend(f"Invent a Dataset: {v['name']}", quote, args.yes):
            continue
        ds = client.datasets.invent(idempotency_key=f"adaption-support-{v['key']}-{args.rows}", **kw)
        invented[v["key"]] = ds.id
        save_state(state)
        print(f"{v['name']}: started {ds.id} ({ds.status})")


if __name__ == "__main__":
    main()
