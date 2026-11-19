"""Turns a ticket conversation into a reply with Adaption's Responses API.

Two retrieval modes:
- "adaption": Adaption's file_search tool over an indexed knowledge-base dataset
  (set KB_DATASET_ID and turn on indexing for it in the Adaption app).
- "local": keyword search in knowledge.py; the hosted model first rewrites the
  question into an English search query so any language works.
- "none": no retrieval, Blueprint only (for evaluating a fine-tuned model closed-book).
"""
import re
import sys
from pathlib import Path

from adaption import AsyncAdaption

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from config import BLUEPRINT  # noqa: E402  same rules the training data was built with

from knowledge import Knowledge, format_context  # noqa: E402

QUERY_INSTRUCTIONS = (
    "Turn the user's latest message into a short English search query (5-12 key terms) for "
    "Adaption Labs' docs, blog and website. Use the earlier conversation to resolve references. "
    "Translate from any language. Keep product, API and method names. Output only the query."
)
STYLE_HINT = (
    "\n\nWrite in the same script the user used: Devanagari for Hindi, Urdu script for Urdu, "
    "Latin script for Hinglish, Roman Urdu, Taglish and Swahili. Mention the Talk to a human "
    "button only when you can't fully answer from the context or the user asks for a person, and "
    "then inside a sentence, not on a line of its own. Don't end a complete answer with an offer of more help. "
    "The user can't see your reference material: never mention 'the context', 'the provided documentation' "
    "or similar; say 'Adaption's docs' or that it isn't public."
)
DEVANAGARI = re.compile(r"[ऀ-ॿ]")
ARABIC = re.compile(r"[؀-ۿ]")


def script_hint(text: str) -> str:
    """The model drifts into Roman Hindi/Urdu; a firm last instruction keeps the user's script."""
    if len(DEVANAGARI.findall(text)) >= 5:
        return ("\n\nThe user wrote in Devanagari script. Write your whole reply in Hindi in Devanagari "
                "script (हिन्दी), not in Roman letters. Only product names, code and URLs stay in English.")
    if len(ARABIC.findall(text)) >= 5:
        return ("\n\nThe user wrote in Urdu script. Write your whole reply in Urdu script (اردو), not in "
                "Roman letters. Only product names, code and URLs stay in English.")
    return ""
SEARCH_HINT = (
    "\n\nSearch the knowledge base before answering anything about Adaption, and answer only "
    "from what you find."
)
MAX_HISTORY = 10
# The search-query rewrite is a tiny task: always use a fast model without hidden reasoning.
QUERY_MODEL = "gemma-4-26b-a4b-it"
# Some hosted models (glm, qwen 397b) reason before answering, and the reasoning counts toward this cap.
MAX_OUTPUT = 2000
FALLBACK = ("Sorry, I couldn't answer that just now. Please try again in a moment, "
            "or use the Talk to a human button below.")


class Answerer:
    def __init__(self, model: str, retrieval: str = "local", kb_dataset_id: str | None = None,
                 query_model: str = QUERY_MODEL):
        # Short timeout, one retry: a Discord user shouldn't wait more than ~30 s.
        self.client = AsyncAdaption(timeout=25, max_retries=1)
        self.query_opts = {"timeout": 10, "max_retries": 0}  # the eval raises these to wait out rate limits
        self.model = model
        self.query_model = query_model
        self.retrieval = retrieval if (retrieval != "adaption" or kb_dataset_id) else "local"
        self.kb_dataset_id = kb_dataset_id
        self.knowledge = Knowledge() if self.retrieval == "local" else None

    async def _query(self, history: list[dict]) -> str:
        convo = "\n".join(f"{m['role']}: {m['content'][:600]}" for m in history[-4:])
        try:
            r = await self.client.with_options(**self.query_opts).responses.create(
                model=self.query_model, instructions=QUERY_INSTRUCTIONS, input=convo,
                max_output_tokens=60, temperature=0)
            return (r.output_text or "").strip() or history[-1]["content"]
        except Exception as e:  # search with the raw message rather than fail
            print(f"query rewrite failed: {e!r}")
            return history[-1]["content"]

    async def reply(self, history: list[dict]) -> tuple[str, list[str]]:
        """history: [{"role": "user"|"assistant", "content": str}], oldest first.
        Returns (reply text, source URLs used)."""
        history = history[-MAX_HISTORY:]
        hint = script_hint(next((m["content"] for m in reversed(history) if m["role"] == "user"), ""))
        try:
            if self.retrieval == "adaption":
                r = await self.client.responses.create(
                    model=self.model, instructions=BLUEPRINT + STYLE_HINT + SEARCH_HINT + hint, input=history,
                    tools=[{"type": "file_search", "dataset_ids": [self.kb_dataset_id]}],
                    max_tool_calls=3, max_output_tokens=MAX_OUTPUT, temperature=0.3)
                return clean(r.output_text), []
            if self.retrieval == "none":  # closed book: how a fine-tuned model is meant to work
                r = await self.client.responses.create(model=self.model, instructions=BLUEPRINT + STYLE_HINT + hint,
                                                       input=history, max_output_tokens=MAX_OUTPUT, temperature=0.3)
                return clean(r.output_text), []
            query = await self._query(history)
            chunks = self.knowledge.search(query)
            instructions = (BLUEPRINT + STYLE_HINT + "\n\nCONTEXT (Adaption's public docs, blog and site; link the most "
                            "relevant URL):\n" + format_context(chunks) + hint)
            r = await self.client.responses.create(model=self.model, instructions=instructions, input=history,
                                                   max_output_tokens=MAX_OUTPUT, temperature=0.3)
            return clean(r.output_text), [c.url for c in chunks if c.url]
        except Exception as e:  # network or API error: never leave the user hanging
            print(f"answer error: {e!r}")
            return FALLBACK, []


def clean(text: str | None) -> str:
    text = (text or "").strip() or FALLBACK
    # Rule 5 of the Blueprint, enforced: the button exists already.
    return re.sub(r"\[\s*Talk to a human\s*\](\([^)]*\))?", "**Talk to a human**", text, flags=re.I)
