"""Step 0: build the public knowledge folder (knowledge/) from the raw scrape.

The raw scrape (data/raw/adaption-dataset/, not in the repo) also holds other
people's posts. This step keeps only sources that are Adaption's own or about
Adaption's own programs, and removes individuals' handles from the curated
knowledge base. Everything downstream (the bot, steps 1-3) reads knowledge/.

    python pipeline/step0_public_knowledge.py
"""
import collections
import json
import re

from config import KNOWLEDGE, RAW

KEEP_TYPES = {
    "docs", "blog_post", "website_page", "llms_txt",             # Adaption's docs and site
    "x_post_official", "x_post_founder", "linkedin_page",        # Adaption's and its founders' posts
    "github_readme", "huggingface_page",                         # Adaption's GitHub and Hugging Face
    "third_party_event_page",                                     # Hackyard's pages for Adaption's challenge
}
# LinkedIn posts: only the founders' own.
KEEP_LINKEDIN = re.compile(r"linkedin\.com/posts/(sararosehooker|sudip-roy)")
# Knowledge-base lines naming or linking private individuals.
DROP_KB_LINE = re.compile(r"publicly thanked|Giagkoudis")
INDIVIDUAL_LINKS = re.compile(r"\s*\[https://x\.com/(shyamol_konwar|asaddogar0)/status/\d+\]")


def main():
    KNOWLEDGE.mkdir(exist_ok=True)
    kept, dropped = collections.Counter(), collections.Counter()
    with open(KNOWLEDGE / "corpus_chunks.jsonl", "w", encoding="utf-8") as out:
        for line in open(RAW / "corpus_chunks.jsonl", encoding="utf-8"):
            r = json.loads(line)
            ok = r["type"] in KEEP_TYPES or (r["type"] == "linkedin_post" and KEEP_LINKEDIN.search(r["source_url"]))
            (kept if ok else dropped)[r["type"]] += 1
            if ok:
                out.write(json.dumps(r, ensure_ascii=False) + "\n")

    kb = (RAW / "knowledge_base.md").read_text(encoding="utf-8")
    lines = [INDIVIDUAL_LINKS.sub("", l) for l in kb.splitlines() if not DROP_KB_LINE.search(l)]
    (KNOWLEDGE / "knowledge_base.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Kept {sum(kept.values())} chunks: {dict(kept)}")
    print(f"Dropped {sum(dropped.values())} chunks: {dict(dropped)}")
    print(f"Knowledge base: {len(kb.splitlines()) - len(lines)} line(s) removed. Wrote {KNOWLEDGE}")


if __name__ == "__main__":
    main()
