"""Step 6: publish the dataset and the LoRA adapter to Hugging Face.

Needs HF_TOKEN in .env (a *write* token from huggingface.co/settings/tokens).
Repos are created **private**; review them on the website, then make them public
there (or rerun with --public).

    python pipeline/step6_publish_hf.py --github <github-user>            # dry run: show what would be uploaded
    python pipeline/step6_publish_hf.py --github <github-user> --upload   # create private repos and upload

Files come from data/hf/dataset/ and data/hf/model/. The placeholders
<hf-user> and <github-user> in the cards are filled in at upload time.
"""
import argparse
import os
import shutil
import tempfile
from pathlib import Path

import common  # noqa: F401  (loads .env)
from config import ROOT as R

DATASET_NAME = "adaption-support-multilingual"
MODEL_NAME = "qwen3.5-9b-adaption-support-lora"


def staged(src: Path, hf_user: str, gh_user: str) -> Path:
    """Copy a folder to a temp dir with the placeholders filled in."""
    dst = Path(tempfile.mkdtemp()) / src.name
    shutil.copytree(src, dst)
    for f in dst.rglob("*"):
        if f.suffix == ".md":
            text = f.read_text(encoding="utf-8")
            f.write_text(text.replace("<hf-user>", hf_user).replace("&lt;hf-user&gt;", hf_user)
                         .replace("<github-user>", gh_user).replace("&lt;github-user&gt;", gh_user), encoding="utf-8")
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--github", required=True, help="GitHub user or org that hosts the bot repo")
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--public", action="store_true", help="create the repos public instead of private")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise SystemExit("HF_TOKEN is missing from .env (a write token from huggingface.co/settings/tokens).")
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    hf_user = api.whoami()["name"]

    jobs = [("dataset", f"{hf_user}/{DATASET_NAME}", R / "data" / "hf" / "dataset"),
            ("model", f"{hf_user}/{MODEL_NAME}", R / "data" / "hf" / "model")]
    for kind, repo, src in jobs:
        files = sorted(p for p in src.rglob("*") if p.is_file())
        size = sum(p.stat().st_size for p in files) / 1e6
        print(f"\n{kind}: {repo} ({'public' if args.public else 'private'}), {len(files)} files, {size:.1f} MB")
        for p in files:
            print(f"  {p.relative_to(src)}")
        if not args.upload:
            continue
        folder = staged(src, hf_user, args.github)
        api.create_repo(repo, repo_type=kind, private=not args.public, exist_ok=True)
        api.upload_folder(folder_path=str(folder), repo_id=repo, repo_type=kind,
                          commit_message="Upload from adaption-discord-support-bot")
        url = f"https://huggingface.co/{'datasets/' if kind == 'dataset' else ''}{repo}"
        print(f"  uploaded: {url}")
    if not args.upload:
        print("\nDry run: nothing uploaded. Add --upload to create the repos and upload.")


if __name__ == "__main__":
    main()
