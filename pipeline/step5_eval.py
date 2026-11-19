"""Step 5: evaluate the support bot per language, scored by an automated judge.

Eval questions: the 36 real public questions people asked about Adaption
(data/seed/eval_real_en.jsonl) plus a sample of the held-out generated set
(eval_en.jsonl). Each is translated into Hindi, Urdu, Tagalog, Swahili and the
four code-switched varieties with Adaption's hosted model (cached), then answered
by the bot and graded against the English reference answer.

Per variety the report shows: accuracy (judge, 0-2 scaled to %), invented facts,
reply in the right language, right script, median length, share with a link,
handoff rate and latency.

    python pipeline/step5_eval.py --dry-run                     # counts + quote, no API calls
    python pipeline/step5_eval.py --base 5 --langs en,hi        # small paid test
    python pipeline/step5_eval.py                               # gemma-4-26b-a4b-it + local retrieval
    python pipeline/step5_eval.py --model qwen3.8-27b
    python pipeline/step5_eval.py --model <fine-tuned> --retrieval none   # closed book

Results resume from data/eval/results_<model>_<retrieval>.jsonl, so a stopped run
picks up where it left off. Report: data/eval/report_<model>_<retrieval>.md.
"""
import argparse
import asyncio
import json
import random
import re
import statistics
import sys
import time
from pathlib import Path

from common import confirm_spend, read_jsonl, write_jsonl
from config import CODE_SWITCH, GEN_MODEL, GEN_PRICE_PER_M, ROOT, SEED

sys.path.insert(0, str(ROOT / "bot"))
from answer import FALLBACK, Answerer  # noqa: E402

from adaption import AsyncAdaption  # noqa: E402

EVAL = ROOT / "data" / "eval"
QUESTIONS = EVAL / "questions.jsonl"

VARIETIES = {
    "en": "English",
    "hi": "Hindi in Devanagari script, as written in India",
    "ur": "Urdu in Urdu (Perso-Arabic) script, as written in Pakistan",
    "tl": "Tagalog, as written in the Philippines",
    "sw": "Swahili, as written in Kenya",
    **{c["key"]: c["desc"] for c in CODE_SWITCH},
}
SCRIPT = {"hi": "deva", "ur": "arab"}  # every other variety is Latin script

TRANSLATE_INSTRUCTIONS = (
    "Rewrite the user's message as a real person would write it in {desc}, in a Discord "
    "support chat. Keep the meaning, keep product, API and method names, code and URLs in "
    "English, keep it casual if the original is casual. Output only the rewritten message."
)
JUDGE_INSTRUCTIONS = (
    "You grade replies from the support bot of the Adaption Labs Discord. You get the user's "
    "question, a reference answer written from Adaption's public docs, and the bot's reply. "
    "The question is written in {desc}; the reply should be in the same language or mix and the "
    "same script (a Devanagari or Urdu-script question answered in Roman letters is language_ok false).\n"
    "Return only JSON: {{\"accuracy\": 0|1|2, \"invented_fact\": true|false, "
    "\"language_ok\": true|false, \"note\": \"<max 15 words, English>\"}}\n"
    "accuracy 2: answers the question correctly and agrees with the reference, or correctly "
    "says the info isn't public and points to a human when the reference does too. "
    "1: partly right, vague, or misses the key point. 0: wrong, off-topic, or a refusal of "
    "an in-scope question. invented_fact: the reply states a specific price, number, date, "
    "name, rule or feature that contradicts the reference or that the reference would not "
    "support. Differences in wording, extra correct detail and links are fine."
)

URL = re.compile(r"https?://")
DEVA = re.compile(r"[ऀ-ॿ]")
ARAB = re.compile(r"[؀-ۿ]")
LATIN = re.compile(r"[A-Za-z]")

# Rough token sizes per call, for the quote only. Judge/translate output includes
# the hidden reasoning deepseek-v4-flash does (~400 tokens per call, measured).
EST = {"translate": (260, 450), "answer_local": (2600, 170), "answer_none": (550, 170), "judge": (900, 500)}
GEN_MAX_TOKENS = 2000


def script_of(text: str) -> str:
    d, a = len(DEVA.findall(text)), len(ARAB.findall(text))
    return "deva" if d >= 5 and d >= a else "arab" if a >= 5 else "latin"


def script_ok(text: str, question: str) -> bool:
    """The reply should use the script the question was written in."""
    # Ignore URLs, code and product names: count letters in the prose only.
    prose = re.sub(r"`[^`]*`|https?://\S+", "", text)
    d, a, l = len(DEVA.findall(prose)), len(ARAB.findall(prose)), len(LATIN.findall(prose))
    want = script_of(question)
    if want == "deva":
        return d > l * 0.6 and d > a
    if want == "arab":
        return a > l * 0.6 and a > d
    return l > (d + a) * 3


def base_questions(n_real: int, n_synth: int) -> list[dict]:
    real_path = SEED / "eval_real_en.jsonl"  # private (other people's posts); absent in the public repo
    real = read_jsonl(real_path)[:n_real] if real_path.exists() else []
    n_synth += n_real - len(real)
    synth = read_jsonl(SEED / "eval_en.jsonl")
    random.Random(7).shuffle(synth)
    out = []
    for i, r in enumerate(real):
        out.append({"qid": f"real{i:03d}", "set": "real", "prompt": r["prompt"], "reference": r["completion"],
                    "category": r.get("category", "")})
    for i, r in enumerate(synth[:n_synth]):
        out.append({"qid": f"held{i:03d}", "set": "heldout", "prompt": r["prompt"], "reference": r["completion"],
                    "category": r.get("category", "")})
    return out


def usage(r) -> tuple[int, int]:
    u = getattr(r, "usage", None)
    return (getattr(u, "input_tokens", 0) or 0, getattr(u, "output_tokens", 0) or 0) if u else (0, 0)


class Run:
    def __init__(self, args):
        self.args = args
        self.client = AsyncAdaption(timeout=120, max_retries=6)
        self.sem = asyncio.Semaphore(args.workers)
        self.gen_tokens = [0, 0]  # translate + judge
        self.gen_cost = 0.0       # from usage.cost, as billed

    async def _gen(self, instructions: str, text: str) -> str:
        # deepseek-v4-flash reasons before answering and the reasoning counts toward
        # max_output_tokens; a tight cap returns an empty, "incomplete" response.
        for _ in range(2):
            r = await self.client.responses.create(model=self.args.judge, instructions=instructions, input=text,
                                                   max_output_tokens=GEN_MAX_TOKENS, temperature=0)
            i, o = usage(r)
            self.gen_tokens[0] += i
            self.gen_tokens[1] += o
            self.gen_cost += getattr(r.usage, "cost", 0) or 0
            if (r.output_text or "").strip():
                return r.output_text.strip()
        return ""

    async def translate(self, q: dict, lang: str) -> dict | None:
        async with self.sem:
            text = await self._gen(TRANSLATE_INSTRUCTIONS.format(desc=VARIETIES[lang]), q["prompt"])
        wrong_script = lang in SCRIPT and script_of(text) != SCRIPT[lang]
        if not text or text.strip() == q["prompt"].strip() or wrong_script:
            print(f"  translation failed: {q['qid']} {lang}{' (wrong script)' if wrong_script else ''}")
            return None  # never fall back to English silently; retried on the next run
        return {**q, "lang": lang, "question": text}

    async def evaluate(self, bot: Answerer, q: dict) -> dict:
        async with self.sem:
            t = time.time()
            reply, sources = await bot.reply([{"role": "user", "content": q["question"]}])
            latency = time.time() - t
            # A bot error is a real failure (score 0); a judge failure is excluded from scores.
            grade = {"accuracy": 0, "invented_fact": False, "language_ok": False, "note": "bot error", "graded": True}
            if reply != FALLBACK:
                raw = await self._gen(JUDGE_INSTRUCTIONS.format(desc=VARIETIES[q["lang"]]),
                                      f"QUESTION:\n{q['question']}\n\nREFERENCE ANSWER:\n{q['reference']}\n\n"
                                      f"BOT REPLY:\n{reply}")
                m = re.search(r"\{.*\}", raw, re.S)
                try:
                    grade = {**json.loads(m.group(0)), "graded": True} if m else {"note": "judge gave no JSON", "graded": False}
                except json.JSONDecodeError:
                    grade = {"note": "judge JSON invalid", "graded": False}
        return {**q, "reply": reply, "sources": sources[:3], "latency": round(latency, 2),
                "words": len(reply.split()), "has_url": bool(URL.search(reply)),
                "handoff": "talk to a human" in reply.lower(), "script_ok": script_ok(reply, q["question"]),
                "accuracy": int(grade.get("accuracy", 0) or 0), "invented_fact": bool(grade.get("invented_fact")),
                "language_ok": bool(grade.get("language_ok")), "note": grade.get("note", ""),
                "graded": grade["graded"], "bot_error": reply == FALLBACK}


def quote(n_translate: int, n_eval: int, retrieval: str, model: str, judge: str) -> str:
    ans = EST["answer_local" if retrieval == "local" else "answer_none"]
    gen_in = n_translate * EST["translate"][0] + n_eval * EST["judge"][0]
    gen_out = n_translate * EST["translate"][1] + n_eval * EST["judge"][1]
    ans_in, ans_out = n_eval * ans[0], n_eval * ans[1]
    gen_cost = (gen_in * GEN_PRICE_PER_M["input"] + gen_out * GEN_PRICE_PER_M["output"]) / 1e6
    ans_cost = (ans_in * GEN_PRICE_PER_M["input"] + ans_out * GEN_PRICE_PER_M["output"]) / 1e6
    return (f"Translations to make: {n_translate} ({judge})\n"
            f"Questions to answer and grade: {n_eval} (answers: {model}, retrieval {retrieval}; judge: {judge})\n"
            f"Estimated tokens: judge/translate ~{gen_in/1e6:.2f}M in, {gen_out/1e6:.2f}M out; "
            f"answers ~{ans_in/1e6:.2f}M in, {ans_out/1e6:.2f}M out\n"
            f"Estimated cost: ~{gen_cost:.3f} credits for judge/translate, ~{ans_cost:.3f} for answers at "
            f"{GEN_MODEL} rates (the API doesn't publish per-model prices; larger models cost more).")


def report(rows: list[dict], tag: str) -> str:
    def pct(xs):
        return f"{100 * sum(xs) / len(xs):.0f}%" if xs else "-"

    ungraded = sum(not r.get("graded", True) for r in rows)
    errors = sum(r.get("bot_error", False) for r in rows)
    lines = [f"# Bot eval: {tag}", "",
             f"{len(rows)} answers. Accuracy: judge score 0-2 scaled to %. Real = public questions people "
             "actually asked; held-out = generated questions never used in training. Judge columns exclude "
             f"{ungraded} answers the judge failed to grade; {errors} bot errors count as wrong.", "",
             "| Variety | n | Accuracy | Real Qs | Invented facts | Right language | Right script | "
             "Median words | With link | Handoff | Median latency |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for lang in [*VARIETIES, "all"]:
        rs = rows if lang == "all" else [r for r in rows if r["lang"] == lang]
        if not rs:
            continue
        g = [r for r in rs if r.get("graded", True)]
        real = [r for r in g if r["set"] == "real"]
        lines.append(
            f"| {'**all**' if lang == 'all' else lang} | {len(rs)} | {pct([r['accuracy'] / 2 for r in g])} | "
            f"{pct([r['accuracy'] / 2 for r in real])} | {pct([r['invented_fact'] for r in g])} | "
            f"{pct([r['language_ok'] for r in g])} | {pct([r['script_ok'] for r in rs])} | "
            f"{statistics.median(r['words'] for r in rs):.0f} | {pct([r['has_url'] for r in rs])} | "
            f"{pct([r['handoff'] for r in rs])} | {statistics.median(r['latency'] for r in rs):.1f}s |")
    worst = sorted((r for r in rows if r.get("graded", True) and (r["accuracy"] == 0 or r["invented_fact"])),
                   key=lambda r: r["lang"])[:15]
    if worst:
        lines += ["", "## Misses to review", ""]
        for r in worst:
            q = r["question"].replace("\n", " ")[:140]
            lines.append(f"- **{r['lang']}** {q} — {r['note']}")
    return "\n".join(lines) + "\n"


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma-4-26b-a4b-it", help="hosted model that answers")
    ap.add_argument("--retrieval", default="local", choices=["local", "none"])
    ap.add_argument("--judge", default=GEN_MODEL, help="hosted model for translation and grading")
    ap.add_argument("--base", type=int, default=60, help="English base questions (36 real first, then held-out)")
    ap.add_argument("--langs", default=",".join(VARIETIES))
    ap.add_argument("--workers", type=int, default=2, help="keep low: the API rate-limits bursts")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    langs = [l for l in args.langs.split(",") if l]
    unknown = set(langs) - set(VARIETIES)
    if unknown:
        sys.exit(f"Unknown varieties: {unknown}. Choose from {list(VARIETIES)}")
    n_real = min(args.base, 36)
    base = base_questions(n_real, args.base - n_real)

    cached = {(q["qid"], q["lang"]): q for q in read_jsonl(QUESTIONS)} if QUESTIONS.exists() else {}
    for q in base:  # English needs no translation
        cached.setdefault((q["qid"], "en"), {**q, "lang": "en", "question": q["prompt"]})
    todo_tr = [(q, l) for q in base for l in langs if (q["qid"], l) not in cached]

    tag = re.sub(r"[^A-Za-z0-9.-]+", "-", f"{args.model}_{args.retrieval}")
    results_path = EVAL / f"results_{tag}.jsonl"
    done = {(r["qid"], r["lang"]): r for r in read_jsonl(results_path)} if results_path.exists() else {}
    wanted = [(q["qid"], l) for q in base for l in langs]
    todo_eval = [k for k in wanted if k not in done]

    print(f"{len(base)} base questions x {len(langs)} varieties = {len(wanted)}; "
          f"{len(wanted) - len(todo_eval)} already graded for {tag}.")
    q = quote(len(todo_tr), len(todo_eval), args.retrieval, args.model, args.judge)
    if args.dry_run:
        print("\n" + q + "\n\nDry run: nothing sent.")
        return
    if (todo_tr or todo_eval) and not confirm_spend("Bot evaluation", q, args.yes):
        return

    run = Run(args)
    if todo_tr:
        print(f"Translating {len(todo_tr)} questions...")
        for new in await asyncio.gather(*(run.translate(q, l) for q, l in todo_tr)):
            if new:
                cached[(new["qid"], new["lang"])] = new
        write_jsonl(QUESTIONS, sorted(cached.values(), key=lambda r: (r["qid"], r["lang"])))
        missing = [k for k in todo_eval if k not in cached]
        if missing:
            print(f"Skipping {len(missing)} questions whose translation failed; rerun to retry them.")
            todo_eval = [k for k in todo_eval if k in cached]

    if todo_eval:
        bot = Answerer(args.model, args.retrieval)
        # Unlike a Discord user, the eval can wait out 429s (the SDK honours retry-after).
        bot.client = AsyncAdaption(timeout=120, max_retries=6)
        bot.query_opts = {"timeout": 120, "max_retries": 6}
        print(f"Answering and grading {len(todo_eval)} questions with {args.workers} workers...")
        EVAL.mkdir(parents=True, exist_ok=True)
        with open(results_path, "a", encoding="utf-8") as f:
            tasks = [asyncio.create_task(run.evaluate(bot, cached[k])) for k in todo_eval]
            for i, t in enumerate(asyncio.as_completed(tasks), 1):
                r = await t
                done[(r["qid"], r["lang"])] = r
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
                if i % 25 == 0:
                    print(f"  {i}/{len(todo_eval)}")
    if todo_tr or todo_eval:
        print(f"Judge/translate: {run.gen_tokens[0]:,} tokens in, {run.gen_tokens[1]:,} out, "
              f"cost {run.gen_cost:.4f} (as reported by the API).")

    rows = [done[k] for k in wanted if k in done]
    out = EVAL / f"report_{tag}.md"
    out.write_text(report(rows, tag), encoding="utf-8")
    print("\n" + out.read_text(encoding="utf-8"))
    print(f"Saved {out}")


if __name__ == "__main__":
    asyncio.run(main())
