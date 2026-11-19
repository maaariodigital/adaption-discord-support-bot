"""Bot settings, read from the project's .env file."""
import os
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "bot"

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass


@dataclass
class Settings:
    discord_token: str
    model: str
    retrieval: str
    kb_dataset_id: str | None
    support_role_id: int | None
    guild_id: int | None


def _int(name):
    v = os.environ.get(name, "").strip()
    return int(v) if v.isdigit() else None


def load_settings(require_discord: bool = True) -> Settings:
    if not os.environ.get("ADAPTION_API_KEY"):
        sys.exit("ADAPTION_API_KEY is missing from .env")
    token = (os.environ.get("DISCORD_TOKEN") or os.environ.get("DISCORD_BOT_TOKEN") or "").strip()
    if require_discord and not token:
        sys.exit("DISCORD_TOKEN is missing from .env. See SETUP.md, step A3.")
    return Settings(
        discord_token=token,
        model=os.environ.get("BOT_MODEL", "gemma-4-26b-a4b-it"),
        retrieval=os.environ.get("BOT_RETRIEVAL", "local"),
        kb_dataset_id=os.environ.get("KB_DATASET_ID") or None,
        support_role_id=_int("SUPPORT_ROLE_ID"),
        guild_id=_int("DISCORD_GUILD_ID"),
    )
