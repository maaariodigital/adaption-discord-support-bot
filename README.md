# Adaption Discord Support Bot

A multilingual, ticket-based support bot for the Adaption Labs Discord, built with Adaption's own
products:

- **Adaptive Data** and **Invent** build the training data.
- **AutoScientist** fine-tunes the model.
- Adaption's **hosted inference** answers in real time.

**Languages:** English, Hindi (India), Urdu (Pakistan), Tagalog (Philippines) and Swahili (Kenya), plus the
code-switched way developers there actually write: Hinglish, Roman Urdu, Taglish and Swahili-English.

| | |
|---|---|
| 🚀 **Set it up** | [SETUP.md](SETUP.md): step-by-step, from a fresh machine to a running bot, plus rebuilding the data and model |
| 📊 **Dataset** | [huggingface.co/datasets/MD1337/adaption-support-multilingual](https://huggingface.co/datasets/MD1337/adaption-support-multilingual) (6,733 rows) |
| 🧠 **LoRA adapter** | [huggingface.co/MD1337/qwen3.5-9b-adaption-support-lora](https://huggingface.co/MD1337/qwen3.5-9b-adaption-support-lora) |
| 📝 **Write-up and product feedback** | [WRITEUP.md](WRITEUP.md) |

## What the bot does

- **Tickets.** `/ticket-panel` posts an **Open a ticket** button. Each ticket is a private thread between the user and the bot.
- **Answers.** Every message gets an answer grounded in Adaption's public docs, blog and site, in the user's language **and script**, with a source link.
- **Feedback and handoff.** Under every answer: 👍 / 👎 (logged) and **Talk to a human**, which pings the support role and pauses the bot until staff run `/resume-bot`.
- **Key safety.** Pasted Adaption API keys are deleted, with a warning to rotate the key.
- **Scope.** Off-topic requests and attempts to override its rules are declined. Unknowns such as unpublished prices or eligibility details are handed to a human instead of guessed.

## Quick start

```bash
git clone https://github.com/maaariodigital/adaption-discord-support-bot.git
cd adaption-discord-support-bot
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # macOS/Linux: .venv/bin/pip
cp .env.example .env                            # add ADAPTION_API_KEY and DISCORD_TOKEN
.venv/Scripts/python bot/ask.py --demo          # try answers in the terminal
.venv/Scripts/python bot/main.py                # run the bot; it prints an invite link
```

Creating the Discord application, intents, IDs and troubleshooting are covered in [SETUP.md](SETUP.md#part-a-run-the-bot).

## How it works

```
question (any language)
  → Adaption hosted model rewrites it into an English search query
  → keyword search over knowledge/ (Adaption's docs, blog, site, official posts)
  → Adaption hosted model answers with the Blueprint rules + retrieved context
  → reply in the user's language and script, with a source link
```

**The Blueprint** ([pipeline/config.py](pipeline/config.py)) is the same set of rules used to build the
training data:
- Reply in the user's language.
- Use only facts from the context.
- Keep Discord-length answers.
- Stay in scope.
- Hand off to a human the right way.

## Repository layout

| Path | What's there |
|---|---|
| `bot/` | Discord bot (`main.py`), answer pipeline (`answer.py`), retrieval (`knowledge.py`), terminal tester (`ask.py`) |
| `knowledge/` | Public knowledge the bot and the pipeline read: a curated knowledge base and 838 chunks from Adaption's docs, blog, site and official posts |
| `pipeline/` | Steps 0–5: build the dataset with Adaptive Data and Invent, train with AutoScientist, evaluate per language |
| `data/`, `models/` | Generated locally; not in the repo. The finished dataset and adapter are on Hugging Face. |

## Data choices

- **Sources.** `knowledge/` holds only Adaption's own material: docs, blog, website, official and founders' posts, GitHub and Hugging Face pages, and Hackyard's pages for Adaption's challenge.
- **Excluded.** Other people's posts, news articles and forum threads are left out of both the knowledge folder and the training data.
- **Hand-written behavior rows** ([pipeline/behavior_rows.py](pipeline/behavior_rows.py)) teach handoff for unknowns, off-topic refusals and API-key hygiene.
- **Held-out eval set.** 8% of the generated questions are held out and never used in training.

## Status

- ✅ Dataset: 6,733 rows in five languages plus four code-switched varieties.
- ✅ Bot running on Adaption hosted inference, evaluated per language (see [WRITEUP.md](WRITEUP.md#results)).
- ⚠️ The first AutoScientist run (Qwen3.5 9B, 2 iterations) reached a win rate of 0.399 against a 0.7 target. The live bot therefore stays on the hosted model with retrieval. The next run would use more iterations and a stronger LoRA recipe.
