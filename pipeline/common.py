"""Helpers shared by the pipeline steps: client, state file, spend confirmation."""
import json
import os
import sys

from config import ROOT, STATE_FILE

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass


def get_client():
    if not os.environ.get("ADAPTION_API_KEY"):
        sys.exit("ADAPTION_API_KEY is not set. Copy .env.example to .env and add your key.")
    from adaption import Adaption
    return Adaption()


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {}


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def confirm_spend(label: str, quote: str, assume_yes: bool) -> bool:
    """Show a credit quote and require an explicit yes before spending."""
    print(f"\n=== {label} ===\n{quote}")
    if assume_yes:
        print("--yes given, proceeding.")
        return True
    if not sys.stdin.isatty():
        print("Not an interactive terminal and no --yes: stopping before spending anything.")
        return False
    try:
        answer = input("Type 'spend' to launch, anything else to stop: ").strip().lower()
    except EOFError:
        print("\nNo input available: stopping before spending anything.")
        return False
    return answer == "spend"


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
