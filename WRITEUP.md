# A multilingual support bot for the Adaption Discord, built with Adaption

**What it is:** a ticket-style Discord support bot for Adaption Labs. Each conversation opens a private
thread, and the bot answers questions about Adaption's products, the company and community programs
(ambassadors, hackathons and challenges, grants). It replies in the user's own language:
English, **Hindi, Urdu, Tagalog and Swahili**, and the code-switched way developers in those
communities actually write (**Hinglish, Roman Urdu, Taglish, Swahili-English**).

**Why these languages:** Adaption's mission is AI that adapts to everyone. These are large developer
communities that most AI support tools serve poorly, and code-switched varieties are almost absent
from public training data.

**How it's built:** almost every step runs on Adaption's own products. The data comes from Adaptive Data
and Invent, the model is trained with AutoScientist, and answers come from Adaption's hosted inference.

---

## What I used, and for what

| Adaption product | How it's used |
|---|---|
| **Responses API** (hosted inference) | `deepseek-v4-flash` drafted ~2,100 grounded Q&A pairs from Adaption's public docs, blog and site. It also cleaned up the final set and grades the eval. `gemma-4-26b-a4b-it` powers the live bot, about 1 s per answer. |
| **Adaptive Data: adapt** | Rewrote the seed with a 5-rule **Blueprint**: reply in the user's language, use only grounded facts, Discord-length answers, scope rules, and how to hand off to a human. Also used: `length: concise` and **deduplication**. |
| **Adaptive Data: localize** | Produced localized (not just translated) copies for hi-IN, ur-PK, tl-PH and sw-KE. |
| **Adaptive Data: Expand** (`augment`) | Generated more questions for thin topics (ambassadors, grants, startups). |
| **Hallucination mitigation** | Used on the programs set, where wrong dates or amounts would do the most harm. |
| **Adaptive Data: evaluation** | First 200-row test: grade **C → B**, score 6.0 → 7.6, +26.7%. With the final concise settings, the scores were part 2: C, −11.7%; programs: C, 0%. My hypothesis is that the grader favours longer answers than a Discord reply should have, but I haven't verified it (see feedback). |
| **Invent a Dataset** | 600 code-switched conversations (150 per variety) that translation can't produce. |
| **AutoScientist** | `recommend_hyperparams` and `list_models` to choose the base model and recipe. Three LoRA fine-tuning runs of `Qwen/Qwen3.5-9B` on all 6,733 rows, 9 iterations in total, each producing a downloaded adapter. Best win rates **0.399, 0.488 and 0.448** against a 0.7 target (see Results). |
| **Inference Stacks** (`file_search`) | The bot has a `BOT_RETRIEVAL=adaption` mode. Its default is a local keyword index, because indexing can only be switched on in the app (see feedback). |

**Final training set: 6,733 rows**

| Variety | Rows |
|---|---|
| English | 2,081 |
| Hindi | 1,016 |
| Urdu | 1,005 |
| Tagalog | 1,026 |
| Swahili | 1,018 |
| Code-switched (all four) | 587 |

- 98% of answers end with a source link.
- Median answer: 80 words.
- 22 hand-written behavior rows teach handoff, off-topic refusal and API-key hygiene.

**Budget:** the whole dataset used about **162 of the 1,000 free credits**.

## Bot features

- **Ticket panel.** `/ticket-panel` posts an *Open a ticket* button. Each ticket becomes a private thread.
- **Language.** Answers in the user's language **and script**: Devanagari, Urdu script, or Latin for the mixed varieties. Product names, code and URLs stay in English.
- **Feedback.** 👍/👎 under every answer, logged for the next round of training data.
- **Talk to a human.** Pings the support role and pauses the bot in that thread; staff resume it with `/resume-bot`.
- **Key safety.** If someone pastes an Adaption API key, the bot deletes the message and tells them to rotate the key.
- **Graceful failures.** Short timeouts and a polite fallback message, so a user is never left waiting.
- **Grounding.** Every answer is grounded in Adaption's public material. Off-topic requests and prompt-injection attempts are declined.

## Data decisions

- **Privacy.** Public posts by other people on X were **excluded from training**, because they are personal data. 36 real questions from those posts are used only as an **eval set**, to measure how the bot does on questions people really asked.
- **Programs data.** Answers about programs are grounded in knowledge-base sections. The answers Expand generated drifted off-topic, so I kept its questions and had Adaptive Data write the answers from the source text.
- **Blueprint retest.** I ran a 200-row test before the full run and changed the Blueprint based on what I saw.
  - Problems in the first test: analyst-style rephrased prompts, refusals of in-scope questions, long answers, and the model writing out "[Talk to a human]" as text.
  - Retest after the changes: **0 refusals, 0 answers over 150 words, 99% with a source link**, by my own checks.
- **Hand-written rows kept as written.** Adapt turned one of them into a false claim about response times, so the originals are used unchanged.

## Results

### Fine-tuning (AutoScientist)

**None of the three runs reached the target or beat the base model.** A win rate of 0.5 means "as good as the untrained model"; the target was 0.7.

| Run | What I changed | Iterations, wall time | Best win rate | Best recipe (from `autoscientist.get`) | Eval loss, first → last eval (`trainer_state.json`) |
|---|---|---|---|---|---|
| 1 | Nothing: AutoScientist's recommendation | 2, 54 min | 0.399 | lr 1e-5, 1 epoch (21 steps), r 16 | 2.31 → 2.21 |
| 2 | More iterations, AutoScientist decides | 4, 1 h 53 min | **0.488** | lr 1e-5, 1 epoch (21 steps), r 8 | 2.31 → 2.28 |
| 3 | Started from lr 1e-4 and 2 epochs (`hyperparams` override) | 3, 1 h 22 min | 0.448 | lr 2e-4, 6 epochs (126 steps), r 48 | 1.20 → 1.03 (lowest 1.00 at step 76) |

**What each run showed:**
- **Runs 1 and 2:** AutoScientist never moved off a learning rate of 1e-5 and a single epoch. Between the two runs' best results, only the LoRA rank and warmup changed, and the eval loss barely moved, so those adapters stay close to the base model.
- **Run 3:** given a stronger starting point, AutoScientist went further on its own, to twice the learning rate, three times the epochs and three times the rank I gave it.
  - The model learned the data far better: eval loss fell from about 2.3 to 1.0.
  - It overfit: eval loss bottomed out at step 76 of 126 and rose after that, while training loss kept falling to 0.47.
  - Even so, its win rate (0.448) was **lower** than run 2's, whose adapter barely changed the model.

**My reading:**
- A much better fit to the data didn't win more comparisons, so the bottleneck isn't the recipe.
- The data teaches short, linked Discord replies that hand off to a human whenever the docs don't cover something. A judge comparing them with the base model's longer, more confident answers likely prefers the longer ones. Adaptive Data's evaluation leaned the same way (see feedback 6).
- A 9B model also can't reliably memorize every fact from 790 knowledge chunks out of 6.7k rows. Fine-tuning changes how the model writes much more than what it knows.
- I can't confirm this without seeing the judge's comparisons (feedback 10).

**What the live bot uses:** it stays on hosted `gemma-4-26b-a4b-it` with retrieval. That design is the right default anyway, since program dates and credit offers change faster than a model can be retrained. The fine-tuning is where the data's style rules would pay off, combined with retrieval rather than replacing it.

### Bot (hosted model + retrieval)

**Setup** (`pipeline/step5_eval.py`):
- 60 English questions: 36 real ones people asked publicly about Adaption, plus 24 held-out generated ones.
- Each was rewritten into all 9 varieties, giving 529 graded answers. 11 translations were skipped after repeatedly coming back in the wrong script.
- Each answer was graded against the English reference by `deepseek-v4-flash`, through Adaption's API.
- The whole eval cost about 0.16 credits.

| Variety | Accuracy | Real questions | Right language | Right script | Median words | With link | Latency |
|---|---|---|---|---|---|---|---|
| English | 76% | 72% | 100% | 100% | 85 | 98% | 1.2 s |
| Hindi | 75% | 71% | 100% | 98% | 93 | 98% | 1.1 s |
| Urdu | 75% | 73% | 100% | 100% | 97 | 98% | 1.2 s |
| Tagalog | 78% | 72% | 93% | 100% | 82 | 93% | 1.4 s |
| Swahili | 69% | 64% | 100% | 100% | 75 | 97% | 1.6 s |
| Hinglish | 65% | 56% | 98% | 100% | 92 | 97% | 1.5 s |
| Roman Urdu | 68% | 61% | 100% | 100% | 89 | 100% | 1.3 s |
| Taglish | 69% | 64% | 97% | 100% | 87 | 97% | 1.2 s |
| Swahili-English | 69% | 60% | 95% | 100% | 74 | 98% | 1.3 s |
| **All** | **71%** | **66%** | **98%** | **100%** | 86 | 97% | 1.3 s |

Accuracy is the judge's 0–2 score scaled to a percentage.

**Model comparison.** I reran the eval with two other hosted models. On the 134 answers all three gave (15 base questions in 9 varieties):

| Model | Accuracy | Flagged invented facts | Median latency | Cost per answer (credits) |
|---|---|---|---|---|
| `gemma-4-26b-a4b-it` (live) | 76% | 34% | 1.4 s | ~0.0003 |
| `gemma-4-31b-it` | 78% | 33% | 2.2 s | ~0.0001 |
| `glm-5.2` | 82% | 65% | 4.3 s | ~0.0017 |

- On the full 539-answer set, gemma-4-31b scored 72% against gemma-4-26b's 70%, with 14 failed answers (rate limits).
- glm-5.2 adds more detail, which helps accuracy but doubles the answers the judge flags for unsupported facts. It is also three times slower and reasons before answering.
- With 134 answers, a few points' difference is within noise, so the bot stays on `gemma-4-26b-a4b-it`.

**What I take from it:**
- **Language is the strong part.** The bot answers in the right language 98% of the time and the right script ~100% of the time.
  - That's after one fix the eval found: Gemma had been answering Devanagari questions in Roman Hindi.
  - Accuracy in the five main languages is close to English (69–78% vs 76%).
- **Code-switched varieties are the gap.** Accuracy drops to 65–69%, and Hinglish on real questions to 56%. That's where more Invent data and native-speaker review would go first.
- **"Invented facts" needs care.** The judge flagged 46% of answers. In a hand check of 12 flagged answers, only 3 had a real error, e.g. "2 VIP passes" where Hackyard's page says "2 VIP swag boxes". The other 9 stated facts that are in Adaption's docs or official posts but not in the short reference answer, so the judge is too strict here.
  - Some references are also out of date. One says there are no docs for a reasoning parameter, but `train_with_reasoning` is documented now.
- **The eval led to two more fixes, made after this run:**
  - Handoff mentions dropped. The bot had ended almost every answer with "use Talk to a human".
  - The bot no longer mentions "the provided context". Up to 11% of answers did.
- **Caveat:** the first ~50 answers used the full scrape as the knowledge index; the rest used the public `knowledge/` folder now in the repo, which has third-party news and posts removed.

## Product feedback

These came up while building, roughly in order of impact.

### Billing and quotes

1. **The free plan's per-launch row cap (~1,000) isn't in the estimate, and a capped run is billed the full quote.**
   - What happened: my 1,894-row adapt was quoted and charged 57 credits but processed 999 rows.
   - Cost: the remaining rows took a second launch (27 credits), about 27 credits more than one uncapped run would have cost.
   - Suggestion: show the cap in `estimate=True` and bill per processed row.
2. **Leaving out `job_specification.max_rows` reuses the previous run's limit.** The docs say it means all rows.
   - What happened: after a 200-row test, the "full" run was quoted at 2 credits. I only caught it because the number looked wrong.
3. **AutoScientist has no cost estimate in the API**, unlike every other paid endpoint. That makes it the hardest step to approve on a budget.

### Progress and timing

4. **Time estimates leave out queue time.**
   - What happened: runs estimated at about 11 minutes took 1–2 hours, and the progress counter sat at 0% the whole time.
   - Suggestion: show the queue position, or an "estimated start" time.

### Evaluation

5. **`get_evaluation` can return an older run's results.**
   - What happened: the retest and the full 1,894-row run on the same dataset both returned the first test's evaluation: the same timestamp, the same `config_source` run ID, and scores identical to the decimal.
   - Effect: there was no way to see whether my Blueprint changes helped. Only datasets with a single run got fresh evaluations.
6. **The evaluation is hard to read for concise, grounded data.**
   - What happened: runs with `length: concise` and `prompt_rephrase` off scored C (completion quality −22% on part 2), although by my own checks they had 0 refusals and 99% sourced answers.
   - Suggestion: say which rubric is used, or let users pass their Blueprint to the grader.
   - Related: with rephrase off, `enhanced_prompt` comes back empty and the localized question is in `prompt`. This isn't documented.

### Models and inference

7. **Deprecated models only show up at `create` time.** `gemma-3-4b-it` was available to plan with, but the deprecation error only came on launch. A deprecated flag in `list_models` would catch this earlier.
8. **`file_search` needs a retrieval index that can only be switched on in the app.** An API flag would let a pipeline set up Inference Stacks from start to finish.
9. **Rate limits aren't documented.** Eight concurrent requests returned 429s with a retry-after of about 37 s. `models.list` also has no per-model prices, so you can't estimate inference cost ahead of time. (`usage.cost` on each response is very useful, though.)
   - **Hidden reasoning eats `max_output_tokens`, and there's no way to turn it off.**
     - What happened: `deepseek-v4-flash` spent all 300 output tokens on reasoning and returned an empty `output_text` with status `incomplete`. My first eval run silently got English text instead of half its translations.
     - Suggestion: a `reasoning` option on `responses.create`, or a reasoning flag in `models.list`.
   - **`gemma-4-26b-a4b-it` drifts out of Devanagari.** It often answers a Hindi question in Roman script despite a system-prompt rule. A firm script instruction at the end of the prompt fixed it.

### AutoScientist results

10. **Hard to see what the win rate means or what the iterations changed.**
    - `autoscientist.get` returns only the best win rate and the best hyperparameters.
    - It doesn't say which model the run was compared against, what judge or prompt set was used, or what each iteration tried.
    - With a 0.399 result, I can't tell whether to change the data, the recipe or the target.
    - Across three runs, the win rate and the eval loss disagreed: run 3 halved the eval loss but scored lower than run 2. Seeing a few judged comparisons would show whether the judge penalizes short answers or handoffs.
    - In runs 1 and 2 (6 iterations), AutoScientist never raised the learning rate above 1e-5 or trained for more than one epoch, although the eval loss barely moved. It only explored stronger recipes after I overrode the starting point, which the docs call "not advised".
11. **The checkpoint download is misnamed.** It's called `best-checkpoint.tgz` but is actually a **zstd**-compressed tar, so `tar -xzf` fails.
12. **The adapter has an empty model card.** Its README is the default Hugging Face template, and `adapter_config.json` names the base as `togethercomputer/Qwen3.5-9B` rather than `Qwen/Qwen3.5-9B`.

### SDK and docs

13. **Small SDK and docs gaps:**
    - Raw-mode upload requires a `completion` column; this isn't documented.
    - Iterating over `models.list()` / `list_models()` yields the response object's fields as tuples instead of the models.
    - The output columns for each recipe combination aren't documented.

### What worked very well

- Localization quality.
- How natural the Invent code-switched conversations read.
- How strongly the Blueprint steers style once it's phrased as concrete rules.
- The estimate-before-spend pattern across the API.

## Next steps

- Rebalance the training data before another AutoScientist run: fewer handoffs in the code-switched rows (about 90% now), and more Invent data for the varieties that score lowest.
- Host an adapter, run the same per-language eval on it with retrieval and closed-book (`--retrieval none`), and switch the bot over only if it beats the hosted model.
- Native-speaker spot checks, starting with Urdu and Swahili.
- Feed 👎-rated answers back into the next Adaptive Data run.
