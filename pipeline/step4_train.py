"""Step 4: train the support model with AutoScientist on the assembled set.

Uses the cleaned training set from step3b_assemble.py --upload, a raw-mode
dataset with prompt/completion columns, so no column mapping is needed.

    python pipeline/step4_train.py            # show recommended recipe, confirm, launch
    python pipeline/step4_train.py --wait     # wait for the run, print win rate, download checkpoint
    python pipeline/step4_train.py --run-name run2 --iterations 4   # a further run; files go to models/run2/

AutoScientist has no credit estimate in the API, so this step shows what will
run and asks for confirmation. Keep max_iterations low for the first run.
"""
import argparse
import json

from common import confirm_spend, get_client, load_state, save_state
from config import BASE_MODEL, MAX_ITERATIONS, ROOT, TARGET_WIN_RATE


def run_key(name: str) -> str:
    return "train_run_id" if name == "run1" else f"train_run_id_{name}"


def wait_and_download(client, state, name):
    run = client.autoscientist.wait_for_completion(state[run_key(name)])
    print(f"Status: {run.status}  iterations {run.iterations_completed}/{run.max_iterations}  "
          f"best win rate {run.best_win_rate} (target {run.target_win_rate})")
    models = ROOT / "models" / name
    models.mkdir(parents=True, exist_ok=True)
    (models / "run.json").write_text(json.dumps(run.model_dump(), indent=2, default=str), encoding="utf-8")
    if run.status != "succeeded" or not run.download_available:
        raise SystemExit(f"No checkpoint to download: {run.error}")
    with client.autoscientist.with_streaming_response.download(run.id) as resp:
        resp.stream_to_file(str(models / "best-checkpoint.tgz"))
    print(f"Saved {models / 'best-checkpoint.tgz'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--iterations", type=int, default=MAX_ITERATIONS)
    ap.add_argument("--general-rows", type=int, default=0,
                    help="general-purpose rows to add against forgetting (extra credits)")
    ap.add_argument("--run-name", default="run1", help="separate state and files per run")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--wait", action="store_true")
    args = ap.parse_args()

    client = get_client()
    state = load_state()
    if args.wait:
        return wait_and_download(client, state, args.run_name)
    if state.get(run_key(args.run_name)):
        raise SystemExit(f"{args.run_name} already launched ({state[run_key(args.run_name)]}). Use --wait or a new --run-name.")
    dsid = state.get("train_dataset_id")
    if not dsid:
        raise SystemExit("Run step3b_assemble.py --upload first.")
    rec = client.datasets.get(dsid)
    kw = {"dataset_id": dsid, "model": args.model}
    if args.general_rows:
        kw["augmentation_general_rows"] = args.general_rows
    hp = client.autoscientist.recommend_hyperparams(**kw)
    quote = (f"Dataset {dsid}: {rec.row_count} rows (+{args.general_rows} general rows)\n"
             f"Model {args.model}, SFT, up to {args.iterations} iteration(s), target win rate {TARGET_WIN_RATE}\n"
             f"Recommended recipe: {hp.model_dump(exclude_none=True)}\n"
             "Cost: not quotable through the API. Check your credit balance in the app before and after.")
    if not confirm_spend("AutoScientist training run", quote, args.yes):
        return
    run = client.autoscientist.create(
        **kw, training_method="instruction", max_iterations=args.iterations,
        target_win_rate=TARGET_WIN_RATE, idempotency_key=f"train-{dsid}-{args.model}-{args.iterations}-{args.run_name}")
    state[run_key(args.run_name)] = run.id
    save_state(state)
    print(f"AutoScientist run {run.id} is {run.status}. It takes hours; check with --wait --run-name {args.run_name}.")


if __name__ == "__main__":
    main()
