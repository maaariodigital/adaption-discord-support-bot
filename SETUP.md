# Setup guide

This walks you through the whole project from a fresh machine:

- **Part A** gets the Discord bot running in your server. It takes about 15 minutes and costs almost nothing.
- **Part B** rebuilds the training dataset and fine-tunes a model with Adaption. It takes a few hours, mostly waiting, and roughly 160 credits plus the AutoScientist run.
- **Part C** evaluates the bot in every language.

You only need Part A to run the bot. The finished dataset and model are already on Hugging Face
(links in the [README](README.md)).

Commands below are written for **Windows** (Git Bash or PowerShell). On macOS/Linux, replace
`.venv/Scripts/python` with `.venv/bin/python`.

---

## What you need

- **Python 3.12** ([python.org](https://www.python.org/downloads/)) and **git**.
- **An Adaption account and API key.** Sign up at [adaptionlabs.ai](https://adaptionlabs.ai), then open
  *Settings → API keys* and create a key. It starts with `pt_live_`. New accounts get free credits.
- **A Discord account** and a server where you have *Manage Server* permission. A new test server is fine.

## Part A: run the bot

### A1. Get the code

```bash
git clone https://github.com/maaariodigital/adaption-discord-support-bot.git
cd adaption-discord-support-bot
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
cp .env.example .env
```

Open `.env` in an editor and paste your Adaption key after `ADAPTION_API_KEY=`.
Never commit `.env`; it's already in `.gitignore`.

### A2. Try the answers without Discord

```bash
.venv/Scripts/python bot/ask.py --demo
```

This asks eight questions: one per language, one off-topic request and one prompt-injection attempt. It
prints each answer with its time and sources. Each answer costs a tiny fraction of a credit. If this works,
your API key and the knowledge base are fine.

Ask your own question:

```bash
.venv/Scripts/python bot/ask.py "Je, AutoScientist inafanya kazi na model gani?"
```

### A3. Create the Discord bot

1. Open the [Discord Developer Portal](https://discord.com/developers/applications) and click **New Application**. Give it a name, e.g. "Adaption Support".
2. Go to the **Bot** tab:
   - Click **Reset Token**, copy the token, and paste it into `.env` after `DISCORD_TOKEN=`. Treat it like a password.
   - Under *Privileged Gateway Intents*, turn on **Message Content Intent** and save. Without it the bot can't read questions.
3. Optional, but recommended:
   - In Discord, open *User Settings → Advanced* and turn on **Developer Mode**.
   - Right-click your server icon → **Copy Server ID**, and paste it after `DISCORD_GUILD_ID=`. Slash commands then appear instantly instead of after up to an hour.
   - Create a role for your support team (e.g. `@Support`). Right-click it in *Server Settings → Roles* → **Copy Role ID**, and paste it after `SUPPORT_ROLE_ID=`. **Talk to a human** pings this role.

### A4. Start the bot and invite it

```bash
.venv/Scripts/python bot/main.py
```

The console shows `Logged in as ...` and an **invite link**.
1. Open the invite link, pick your server and click **Authorize**. The link already requests the right permissions: send messages, create and manage private threads, manage messages for deleting leaked API keys, and read history.
2. In the channel where people should open tickets, type `/ticket-panel`. You need *Manage Server* for this. The bot posts a panel with an **Open a ticket** button.

The bot runs as long as this terminal stays open. To keep it online, run it on a small always-on server
or VM with the same commands.

### A5. Use it

- **Open a ticket** creates a private thread with you and the bot. Ask anything about Adaption, in any of the supported languages. Follow-up questions in the thread use the conversation history.
- Under each answer:
  - **👍 / 👎** are logged to `data/bot/feedback.jsonl`, which is useful for the next round of training data.
  - **Talk to a human** pings the support role, pauses the bot in that ticket, and marks the thread with 🙋. Staff type `/resume-bot` in the thread to turn the bot back on.
- **Close ticket** (owner or staff) archives and locks the thread.
- If someone pastes an Adaption API key, the bot deletes the message and tells them to rotate the key.

### A6. Options

| `.env` setting | Default | What it does |
|---|---|---|
| `BOT_MODEL` | `gemma-4-26b-a4b-it` | Adaption hosted model for answers. Any ID from `client.models.list()` works. |
| `BOT_RETRIEVAL` | `local` | `local`: keyword search over `knowledge/`. `adaption`: Adaption's `file_search` over `KB_DATASET_ID`, which needs indexing switched on for that dataset in the Adaption app. |

### Troubleshooting

| Symptom | Fix |
|---|---|
| `PrivilegedIntentsRequired` on start | Turn on **Message Content Intent** (step A3) and restart. |
| `/ticket-panel` doesn't show up | Set `DISCORD_GUILD_ID` and restart, or wait up to an hour for global commands. Also check that the invite included `applications.commands` (the printed link does). |
| Bot replies "Sorry, I couldn't answer that just now" | Usually Adaption's rate limit (HTTP 429) or a network error; the console shows which. Wait a minute. |
| `DISCORD_TOKEN is missing` | Paste the token into `.env` (`DISCORD_BOT_TOKEN` works too). |
| The bot can't create threads | Give its role *Create Private Threads* and *Send Messages in Threads* in that channel. |

## Part B: rebuild the dataset and fine-tune (optional)

Every step that spends credits first asks Adaption for a quote, shows it, and waits for you to type
`spend`. Credit figures below are what this project actually used.

| Step | Command | What it does | Credits |
|---|---|---|---|
| 0 | `python pipeline/step0_public_knowledge.py` | Only needed if you have your own raw scrape in `data/raw/`; `knowledge/` is already in the repo | free |
| 1 | `python pipeline/step1_build_seed.py --dry-run`, then without `--dry-run` | Drafts ~2,100 English Q&A pairs from `knowledge/` with `deepseek-v4-flash`, and holds 8% out for evaluation | ~0.1 |
| 2 | `python pipeline/step2_adapt.py --max-rows 200` | Optional test: adapt + localize a 200-row sample | ~6 |
| 2 | `python pipeline/step2_adapt.py`, then `--download` | Adaptive Data: Blueprint, dedup, localize into hi-IN, ur-PK, tl-PH, sw-KE | ~57 |
| 2 | `python pipeline/step2_adapt.py --part2`, then `--download --part2` | The rows the first launch didn't reach (the free plan caps a launch at ~1,000 rows) | ~27 |
| 2b | `python pipeline/step2b_programs.py`, then `--download` | Expand thin program topics, then adapt with hallucination mitigation | ~10 |
| 3 | `python pipeline/step3_invent_codeswitch.py`, then `--download` | Invent: 150 Hinglish, Roman Urdu, Taglish and Swahili-English rows each | ~60 |
| 3b | `python pipeline/step3b_assemble.py --upload` | Cleans and merges everything into `data/final/train.jsonl` and uploads it | ~0.02 |
| 3c | `python pipeline/step3c_grounded.py --dry-run`, then `--upload` | Optional: rebuilds the set with the docs the bot would retrieve inside each prompt, drops rows whose facts aren't in `knowledge/`, and re-answers the code-switched rows from those docs. Train on it with `step4_train.py --dataset grounded` | ~0.06 |
| 4 | `python pipeline/step4_train.py`, then `--wait` | AutoScientist: LoRA fine-tune of `Qwen/Qwen3.5-9B`, then downloads the adapter | no quote in the API |
| 6 | `python pipeline/step6_publish_hf.py --github <you> --upload` | Publishes the dataset and adapter to Hugging Face as private repos. Needs `HF_TOKEN` (a write token) in `.env`; the cards are staged in `data/hf/`. | free |

Prefix each command with `.venv/Scripts/` as in Part A. Runs take **1–2 hours each, mostly queue
time**, even when Adaption estimates minutes. The `--download` commands tell you if a run is still going,
so it's safe to rerun them. Run IDs are kept in `data/work/state.json`, so you can stop and resume any time.

**Unpacking the adapter:** the downloaded `models/run1/best-checkpoint.tgz` is a *zstd*-compressed tar, despite
the `.tgz` name. Unpack it with:

```bash
.venv/Scripts/python -c "import tarfile,zstandard; tarfile.open(fileobj=zstandard.ZstdDecompressor().stream_reader(open('models/run1/best-checkpoint.tgz','rb')), mode='r|').extractall('models/run1/lora', filter='data')"
```

**Using the adapter:** it's a standard PEFT LoRA for `Qwen/Qwen3.5-9B`. Load it with `transformers` + `peft`,
or serve it with vLLM (`--enable-lora`). Then point `BOT_MODEL` at an OpenAI-compatible endpoint. The bot
currently calls Adaption's Responses API, so that last step needs a small change in `bot/answer.py`.

## Part C: evaluate the bot (optional)

```bash
.venv/Scripts/python pipeline/step5_eval.py --dry-run
.venv/Scripts/python pipeline/step5_eval.py
```

What it does:
1. Translates held-out questions into all 9 varieties.
2. Has the bot answer each one.
3. Grades every answer against the reference with a hosted judge model.
4. Writes `data/eval/report_<model>_<retrieval>.md`, with accuracy, invented facts, right language, right script, length, links and latency for each variety.

The full run costs about 0.16 credits and takes 30–40 minutes. Workers are kept low to stay under Adaption's
rate limit. Compare models with `--model <id>`, or test a fine-tuned model without retrieval with
`--retrieval none`.
