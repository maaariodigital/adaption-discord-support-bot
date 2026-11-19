"""Local keyword search over Adaption's public docs, blog, site and the curated
knowledge base. Used when the Adaption file_search index isn't set up.

Questions can come in any language, so the answer pipeline first turns them into
an English search query (see answer.py); the index itself is English.
"""
import json
import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

RAW = Path(__file__).resolve().parent.parent / "knowledge"  # public subset, see pipeline/step0
# Third-party posts and Discord metadata hold other people's data; keep them out.
EXCLUDE_TYPES = {"x_post_public", "discord_public_metadata"}
KB_BOOST = 1.3  # the curated knowledge base is the most reliable source
MAX_CHARS = 1500
STOP = set("a an the and or of to in on for is are was be with by at as it this that how what do does can i my we our you your from".split())


@dataclass
class Chunk:
    title: str
    url: str
    text: str
    kb: bool = False


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9_.]+", text.lower()) if t not in STOP and len(t) > 1]


def _load_corpus() -> list[Chunk]:
    chunks = []
    for line in open(RAW / "corpus_chunks.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r.get("type") in EXCLUDE_TYPES:
            continue
        body = re.sub(r"^\[Source:[^\]]*\]\n", "", r["text"])
        body = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"\1", body)  # drop inline link URLs
        if len(body.strip()) < 40:
            continue
        chunks.append(Chunk(title=r.get("title") or "", url=r.get("source_url") or "", text=body.strip()))
    return chunks


def _load_kb() -> list[Chunk]:
    kb = (RAW / "knowledge_base.md").read_text(encoding="utf-8")
    chunks = []
    for section in re.split(r"(?m)^## ", kb)[1:]:
        title, _, body = section.partition("\n")
        if title.startswith("8."):
            continue
        # Split long sections at ### headings so each chunk stays focused.
        parts = re.split(r"(?m)^### ", body)
        for n, part in enumerate(parts):
            if len(part.strip()) < 40:
                continue
            sub = part.split("\n", 1)[0].strip() if n else ""  # parts after the first start with a ### title
            urls = re.findall(r"https?://[^\s\])]+", part)
            chunks.append(Chunk(title=f"{title.strip()} {sub}".strip(), url=urls[0] if urls else "",
                                text=part.strip()[:4000], kb=True))
    return chunks


class Knowledge:
    def __init__(self):
        self.chunks = _load_kb() + _load_corpus()
        self.bm25 = BM25Okapi([_tokens(f"{c.title} {c.text}") for c in self.chunks])

    def search(self, query: str, k: int = 6) -> list[Chunk]:
        scores = self.bm25.get_scores(_tokens(query))
        ranked = sorted(range(len(self.chunks)),
                        key=lambda i: scores[i] * (KB_BOOST if self.chunks[i].kb else 1.0), reverse=True)
        out, seen = [], set()
        for i in ranked:
            c = self.chunks[i]
            key = (c.url, c.text[:200])
            if scores[i] <= 0 or key in seen:
                continue
            seen.add(key)
            out.append(c)
            if len(out) == k:
                break
        return out


def format_context(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[{i}] {c.title} | {c.url}\n{c.text[:MAX_CHARS]}" for i, c in enumerate(chunks, 1))
