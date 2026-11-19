# Adaption Labs knowledge base (public sources only)

All facts were retrieved on **2026-11-17** from the public URLs shown in brackets. Figures such as "82%" and "35%" come from Adaption's own evaluations, and Adaption itself frames them as averages, not guarantees. Full source text is in `corpus.jsonl` and `raw/`.

---
## 1. Company

- **Name / legal:** Adaption, Inc. ("Adaption Labs"), operating under the marketing name "Adaption". Address in contest terms: 2261 Market Street STE 46020, San Francisco, CA 94114. [https://adaptionlabs.ai/autoscientist-challenge]
- **What it does:** builds continual-learning systems that let organizations train, customize, and own their AI. [https://adaptionlabs.ai/llms.txt]
- **Three pillars:** Adaptive Data (Adaptive Data platform plus Invent a Dataset), Adaptive Intelligence (incl. AutoScientist), and Adaptive Interfaces (Visualize is the first release). [https://adaptionlabs.ai/llms.txt]
- **Founders:** Sara Hooker, co-founder & CEO (formerly VP of AI Research at Cohere, Google Brain/DeepMind; author of "The Hardware Lottery"), and Sudip Roy, co-founder & CTO (formerly director of inference computing at Cohere). [https://fortune.com/2026/02/04/adaption-labs-50-million-seed-funding-emergence-captial-sara-hooker-sudip-roy-ai-models-that-learn-on-the-fly/] [https://techcrunch.com/2026/05/13/adaption-aims-big-with-autoscientist-an-ai-tool-that-helps-models-train-themselves/]
- **Funding:** $50M seed announced Feb 4, 2026, led by Emergence Capital Partners, with Mozilla Ventures, Fifty Years, Threshold Ventures, Alpha Intelligence Capital, E14 Fund, and Neo. Valuation was not disclosed. [Fortune, above] [https://www.wsgr.com/en/insights/wilson-sonsini-advises-adaption-labs-on-dollar50-million-seed-funding.html] [https://x.com/adaption_ai/status/2019033195178799507]
- **HQ:** San Francisco (SF HQ referenced by Sara Hooker on X). [https://x.com/sarahookr/status/2121708766853054897]
- **Team news:** Hugo Larochelle joined as Scientific Lead (Jul 2, 2026). [https://adaptionlabs.ai/blog/hugo-larochelle] Yuntian Deng joined to shape research on adaptive interfaces (Oct 30, 2026). [https://x.com/sarahookr/status/2116225780334408071]
- **Partnerships:** Together AI (Apr 30, 2026) [https://adaptionlabs.ai/blog/together-ai]; NCS, sovereign AI in Asia-Pacific (Jun 22, 2026) [https://adaptionlabs.ai/blog/ncs-partner-sovereign-ai-asia-pacific]; AI Singapore pilot (Aug 4, 2026) [https://adaptionlabs.ai/blog/ai-singapore-sovereign-ai-pilot]; Hugging Face integration ("Export to Hugging Face with one click", Sep 8, 2026) [https://x.com/adaption_ai/status/2097294140144853313]; Kaggle (Mar 24, 2026) [https://x.com/adaption_ai/status/2036412716009791520]; plugins for ChatGPT/Codex (Nov 4, 2026) [https://x.com/adaption_ai/status/2118010973432725976].
- **Perk:** the "adaption passport", $2,000 per year toward a trip to a country the employee has never visited. [https://x.com/sarahookr/status/2122448754821468242] [https://observer.com/2026/02/sara-hooker-raises-50m-to-challenge-a-i-s-conventional-wisdom-interview/]

## 2. Careers (snapshot as of 2026-11-17) [https://adaptionlabs.ai/careers]
11 open roles. Modelling: Modelling Resident (contract, remote), Research Scientist, Senior Research Scientist (remote, many countries). Platform: Agent Systems Engineer, Inference Engineer, Platform Engineer for APIs and Security (SF, hybrid). Marketing: **Developer Community Manager (Contract, remote: US/UK/PL/MX/PT/ES/IE/DE/CA)**, Founding Growth Marketer, ML Storyteller, Product and Content Marketer. Operations: Head of Operations (SF). Applications go through Ashby links on the careers page.

## 3. Products

### Adaptive Data [https://adaptionlabs.ai/adaptive-data] [https://docs.adaptionlabs.ai/adaptive-data/overview/]
- Optimizes, expands, and generates AI training data across **242 languages and localizations**, for text, documents, and images. Adaption reports an **82% average data-quality increase** in early deployments (an average, not a guarantee). [https://adaptionlabs.ai/llms.txt]
- Outputs **instruction datasets** (prompt/completion for SFT) and **preference pairs** (chosen/rejected for DPO).
- Six stages: Import, Map columns, Apply recipes (dedupe, rephrase, reasoning traces), Expand (augment/translate/localize), Brand controls (length, safety, grounding/hallucination mitigation, Blueprint), Adapt and evaluate.
- **Inputs:** CSV, JSON, JSONL, Parquet, PDF, DOCX, PPTX, XLSX, HTML, ZIP, TXT. Can also import from Hugging Face or Kaggle URLs (Kaggle credentials go in API-key settings).
- **Column roles:** `prompt`, `completion`, `chat` (list of {role, content}), `context` (list), `image`, `universal_prompt`. If you only have prompts, it generates completions. If you only have completions, it synthesizes prompts. [https://docs.adaptionlabs.ai/adaptive-data/select-columns/]
- **Export:** `download(file_format=csv|json|jsonl|parquet)`, default csv.
- Feature launches: Blueprint (Mar 17, 2026) [https://adaptionlabs.ai/blog/blueprint]; API & Python SDK (Apr 9) [https://adaptionlabs.ai/blog/adaptive-data-api-and-python-sdk]; Expand Your World, scaling from as few as 10 examples to up to 2,420 across 242 languages (Apr 14) [https://adaptionlabs.ai/blog/expand-your-world]; Forge, turning unstructured docs into datasets (Apr 23) [https://adaptionlabs.ai/blog/forge]; Multimodal (Jun 2) [https://adaptionlabs.ai/blog/adaptive-data-multimodal]; Expand, "Start with 10 data points. Leave with 100K." (Nov 12) [https://adaptionlabs.ai/blog/expand].

### Invent a Dataset [https://adaptionlabs.ai/invent-a-dataset] [https://adaptionlabs.ai/blog/introducing-invent-a-dataset]
- Builds a training-ready dataset from a description of the desired behavior, with no seed data, schema, or labeling guide. Launched Sep 3, 2026. Invent API launched Sep 15, 2026. Commercial use of Invent API datasets for downstream training is permitted. [https://adaptionlabs.ai/llms.txt]
- Technical report (Oct 1, 2026): Invent API datasets were 19% to 55% more diverse than every model tested, with about 17% higher quality than the strongest baselines (Adaption's own evaluation). [https://adaptionlabs.ai/blog/measuring-dataset-generation-abilities-with-zero-seed-data]
- Invent pricing is "still provisional" per the API docs. [https://docs.adaptionlabs.ai/api/resources/datasets/]

### AutoScientist [https://adaptionlabs.ai/auto-scientist] [https://docs.adaptionlabs.ai/autoscientist/overview/]
- Launched May 13, 2026. Automates the research loop behind model training and alignment: data optimization, then training (SFT or DPO; LoRA or full), then evaluation vs. `target_win_rate`, then hyperparameter adjustment, repeating until the target is hit or `max_iterations` is reached. The download is the **best** checkpoint (tar archive). [https://adaptionlabs.ai/blog/autoscientist]
- Claims (Adaption's own evaluations): outperformed configurations from its in-house research staff by an average of 35%, with win rates rising from 48% to 64%. [https://adaptionlabs.ai/blog/autoscientist] Medical long-context reasoning: correctness up from 58.1% to 75.2%. [https://adaptionlabs.ai/blog/self-improving-intelligence-for-long-context-medical-reasoning]
- Free for 30 days after launch ("For the next 30 days, AutoScientist is on us", May 20, 2026). This was a time-limited offer. [https://x.com/adaption_ai/status/2057068838340936026] [TechCrunch, above]
- Variants and extensions: Tiny AutoScientist for models under 10B parameters (0.8B to 8B) (Jun 10, 2026) [https://adaptionlabs.ai/blog/tiny-autoscientist]; multimodal (Jul 16) [https://adaptionlabs.ai/blog/multimodal-autoscientist]; AutoScientist API (Jul 28) [https://adaptionlabs.ai/blog/autoscientist-api]; AutoScientist Alignment, which configures RL recipes automatically (Aug 25) [https://x.com/adaption_ai/status/2092220458682732753].
- **Supported base models** include Gemma 3/4, gpt-oss 20B/120B, Llama 3.2/3.3/4 Scout, Mistral 7B, Mixtral 8x7B, Nemotron 3, Qwen 3.5/3.6, plus VLMs. **Minimum effective rows are 1,000 (10,000 for some models), and DPO needs at least 12,000 rows.** If you omit `model`, AutoScientist picks one. [https://docs.adaptionlabs.ai/autoscientist/supported-models/]
- Runbook recommendation: aim for at least 20,000 domain rows plus 40% general rows, via augmentation. [https://docs.adaptionlabs.ai/runbooks/train-with-autoscientist/]

### Visualize [https://adaptionlabs.ai/blog/visualize]
- Launched Oct 15, 2026, as the first Adaptive Interfaces release. Turns plain-language questions about a dataset into interactive charts linked to the rows behind each pattern.

### Inference / Responses API [https://docs.adaptionlabs.ai/inference-stacks/models/]
- Hosted models priced in credits per 1M tokens (e.g., gemma-4-31b-it: 0.09 input / 0.34 output credits). See the docs for the full table.

### Developer tools [https://docs.adaptionlabs.ai/] [https://github.com/adaptionlabs]
- Python SDK: `pip install adaption`. API keys come from Settings, then API keys. Use the env var `ADAPTION_API_KEY`. An async client (`AsyncAdaption`) is available. API base: api.prod.adaptionlabs.ai.
- MCP server, plus plugins for Claude Code, Cursor, Codex, Hermes, and OpenCode (OAuth sign-in). Runbooks are available as Colab notebooks (github.com/adaptionlabs/adaption-runbooks).
- Open dataset: huggingface.co/datasets/adaptionlabs/preference-pairs (140,582 preference pairs, 6 languages, 8 domains, with reasoning traces).

### Enterprise [https://adaptionlabs.ai/enterprise]
- Adaptive Data plus AutoScientist on SOC 2 certified infrastructure, 242 languages, 13+ industries. Contact sales: https://adaptionlabs.ai/book-a-call

## 4. Pricing & credits
- Usage is credit-based. You can estimate cost with `datasets.adapt(..., estimate=True)`. **Unused credits do not expire.** New monthly credits are added on top of the existing balance. [https://docs.adaptionlabs.ai/resources/faq/]
- Expand costs 1 credit per 100 rows (rounded up). Multimodal adapt costs 10 credits per 100 output rows. [https://docs.adaptionlabs.ai/adaptive-data/configure-adaptive-data/] [https://docs.adaptionlabs.ai/api/resources/datasets/]
- A "Plus" plan exists (Adaptive Data Plus, given to startups and challenge participants). **No public price list for plans or credits was found** (the app pricing page is behind login). [https://adaptionlabs.ai/blog/adaption-for-startups]

## 5. Support & community channels
- **Discord (official):** https://discord.gg/sHhG8kwVav. It's linked in the site footer and on the docs Support page. Server "adaption": "Building Intelligence that Evolves with the World. This is our discord to connect with researchers, builders, communities who care about fundamental innovation to build a new era of intelligence." About 2,450 members (about 178 online) on 2026-11-17. It's a Community server, opened publicly on Jul 22, 2026. [https://x.com/adaption_ai/status/2079899261873897719] [Discord public invite API]
- Support guidance: when asking for technical help, include a problem statement, dataset/run IDs, timestamps, and errors. Never post API keys or private data. [https://docs.adaptionlabs.ai/resources/support/]
- **Email:** support@adaptionlabs.ai (account, billing, sensitive); team@adaptionlabs.ai (contest terms questions). [docs support page] [https://adaptionlabs.ai/autoscientist-challenge]
- Social: X @adaption_ai, LinkedIn company/adaption-labs, YouTube @Adaption-AI. [https://adaptionlabs.ai/]

## 6. Community programs

### Ambassadors program (launched Apr 21, 2026) [https://adaptionlabs.ai/blog/ambassador-program] [https://adaptionlabs.ai/ambassadors-code-of-conduct]
- Six-month program for builders, community organizers, and "ecosystem firestarters". Applications are open worldwide, reviewed weekly, and countries not yet represented get priority. Apply at https://adaptionlabs.ai/ambassadors-application
- Benefits: early access and beta features, a private Discord channel with leadership, roadmap input, AMAs and panels, mentorship and skill sessions, an official title and badge, exclusive gear, credits, and grants for hosting local meetups and workshops.
- Code of conduct: at least 6 months of active engagement. Events must carry Adaption branding and be promoted on LinkedIn and X. Event grants are available up to once a month via the private Discord channel and must be approved first. Ambassadors must give honest feedback, keep non-public info confidential, not misuse credits, disclose other ambassador programs, and be 18 or older.

### Research Grant Program (launched Mar 10, 2026) [https://adaptionlabs.ai/grants] [https://adaptionlabs.ai/blog/adaption-research-grant-program]
- Complimentary platform access for academic researchers worldwide. Projects must advance open science or the public good (e.g., publication). Rolling applications. A planned annual symposium for grantees. Apply at https://adaptionlabs.ai/research-grant-program-apply

### Public Sector Grant (launched May 18, 2026) [https://adaptionlabs.ai/public-sector-grant] [https://adaptionlabs.ai/blog/publicsector-grant]
- For public agencies, universities, and civic organizations facing multilingual, multi-jurisdictional challenges. Submit a proposal at https://adaptionlabs.ai/publicsectorgrant-application

### Adaption for Startups (launched Mar 31, 2026) [https://adaptionlabs.ai/adaption-for-startups] [https://adaptionlabs.ai/blog/adaption-for-startups]
- Early-stage startups get the Adaptive Data Plus plan, free credits, and access to a builder community. Rolling applications, worldwide. Apply at https://adaptionlabs.ai/adaption-for-startups/apply . Public acceptances include Crane AI Labs and OrganoBit. [LinkedIn posts in corpus]

### Challenges / hackathons
- **Uncharted Data Challenge** (Apr 10 to about May 1, 2026): two weeks, $20,000 pool ($12k/$5k/$3k), open-source datasets for underserved domains or under-resourced languages, published to Kaggle or HF crediting Adaptive Data. Hosted on Kaggle. [https://adaptionlabs.ai/blog/the-uncharted-data-challenge] [https://adaptionlabs.ai/uncharted-data-challenge-terms] [https://www.kaggle.com/competitions/the-uncharted-data-challenge]
- **AutoScientist Challenge** (sign-ups Jun 8; Part 1 Jun 8 to 22; Part 2 Jun 23 to Jul 6, 2026; winners Jul 10): $50,000 pool ($4,000 × 10 first prizes, $1,000 × 10 runners-up, 30 honorable mentions), 10 categories, 1,000 credits per accepted participant. Judged on held-out test sets. Datasets and weights had to be open-sourced to HF and Kaggle. Quebec residents, under-18s, and Adaption staff were excluded. [https://adaptionlabs.ai/autoscientist-challenge] [https://adaptionlabs.ai/blog/autoscientist-challenge] Leaderboard: https://adaptionlabs.ai/auto-scientist-leaderboard
- **Invent a Dataset challenge**: a public winner post exists (Oct 21, 2026), but no official page was found.
- **Hackyard × Adaption Dataset Challenge** (active now): online, Nov 16 to Dec 6, 2026 (submissions close 11:59 PM IST). Onboarding Nov 18, 9:00 AM EST. Teams of 1 to 4, ages 18+. $500 credits for every registered builder (Hackyard posts also mention 3 months of Plus). The top 3 get +$1,000 credits per member and swag; 1st also gets $100 cash and 2 VIP swag boxes. Judging Dec 7 to 13. Submit on Unstop or at app.hackyard.org/e/adaption-labs. [https://hackyard.org/events/adaption-labs/faq] [https://hackyard.org/events/adaption-labs/prizes] [https://hackyard.org/events/adaption-labs/submit]
- **Other partner hackathons** (from public X posts): 3SVK Season 4 "Powered by Adaption Labs" on Unstop ($500 credits for all) [https://x.com/3SVKOfficial/status/2118711863366300094]; Hackyard Build 2026 at IIT Guwahati; build54Africa (Oct 2026) [https://x.com/build54Africa/status/2114482026644717910]; a HackIndia hackathon channel exists in the Discord (invite metadata).

### Events (examples)
- Sara Hooker spoke at the tokens&ai Backprop Summit on Nov 17, 2026 [https://x.com/adaption_ai/status/2122362845895495934]; Sudip Roy at Snorkel's Frontier Data Summit on Oct 8, 2026 [https://x.com/adaption_ai/status/2105266424130539539]; Field Notes ep. 1 registration page [https://adaptionlabs.ai/fieldnotes-ep1]; India AI Impact Summit, New Delhi, Feb 2026 [https://x.com/adaption_ai/status/2021606941827514876]. More are in `corpus.jsonl` (type `x_post_official`).

## 7. Blog / launch timeline (dates from the posts)
| Date | Title | URL |
|---|---|---|
| 2026-02-24 | Adaptive Data: Designed for Change | https://adaptionlabs.ai/blog/adaption-launches-adaptive-data-beta |
| 2026-03-10 | Adaption Research Grant Program: Where Curiosity Pushes Breakthroughs | https://adaptionlabs.ai/blog/adaption-research-grant-program |
| 2026-03-17 | Blueprint: A Specification Layer for Adaptive Data | https://adaptionlabs.ai/blog/blueprint |
| 2026-03-31 | Introducing Adaption for Startups: AI That's Yours to Shape | https://adaptionlabs.ai/blog/adaption-for-startups |
| 2026-04-02 | The Uncharted Data Challenge | https://adaptionlabs.ai/blog/the-uncharted-data-challenge |
| 2026-04-09 | The Adaptive Data API and Python SDK | https://adaptionlabs.ai/blog/adaptive-data-api-and-python-sdk |
| 2026-04-14 | Expand Your World | https://adaptionlabs.ai/blog/expand-your-world |
| 2026-04-21 | The Adaption Ambassador Program | https://adaptionlabs.ai/blog/ambassador-program |
| 2026-04-23 | Forge: Turn Unstructured Documents Into AI Ready Datasets | https://adaptionlabs.ai/blog/forge |
| 2026-04-30 | Build Models Shaped By Adaptive Data and Together AI | https://adaptionlabs.ai/blog/together-ai |
| 2026-05-13 | AutoScientist: Automating the Science of Model Training | https://adaptionlabs.ai/blog/autoscientist |
| 2026-05-18 | Introducing the Public Sector Grant | https://adaptionlabs.ai/blog/publicsector-grant |
| 2026-06-02 | Beyond Text: Adaptive Data for the Multimodal Era | https://adaptionlabs.ai/blog/adaptive-data-multimodal |
| 2026-06-05 | AutoScientist Challenge: Bring Frontier AI to the World | https://adaptionlabs.ai/blog/autoscientist-challenge |
| 2026-06-10 | Tiny AutoScientist: Supersized Intelligence for Small Models | https://adaptionlabs.ai/blog/tiny-autoscientist |
| 2026-06-22 | Adaption and NCS Partner to Bring Sovereign AI to Enterprise Teams Across Asia-Pacific | https://adaptionlabs.ai/blog/ncs-partner-sovereign-ai-asia-pacific |
| 2026-07-02 | Hugo Larochelle Joins Adaption as Scientific Lead | https://adaptionlabs.ai/blog/hugo-larochelle |
| 2026-07-16 | AutoScientist Goes Multimodal: Automate AI Training Across Text and Images | https://adaptionlabs.ai/blog/multimodal-autoscientist |
| 2026-07-28 | AutoScientist API: Automated Model Training | https://adaptionlabs.ai/blog/autoscientist-api |
| 2026-08-04 | Adaption and AI Singapore Pilot to Improve Model Training Efficiency in Southeast Asia | https://adaptionlabs.ai/blog/ai-singapore-sovereign-ai-pilot |
| 2026-08-13 | Better Agent Memory Starts Before Retrieval | https://adaptionlabs.ai/blog/agent-memory-write-time |
| 2026-09-03 | Invent a Dataset: Data Generation for Custom Models | https://adaptionlabs.ai/blog/introducing-invent-a-dataset |
| 2026-09-09 | A Better Harness Can Unlock Smaller Models | https://adaptionlabs.ai/blog/a-better-harness-can-unlock-smaller-models |
| 2026-09-11 | RAG vs. Fine-Tuning vs. Continual Learning: Which Approach Is Right for Your AI System? | https://adaptionlabs.ai/blog/rag-vs-fine-tuning-vs-continual-learning |
| 2026-09-21 | What Is Continual Learning in AI? A Guide for Enterprise Teams | https://adaptionlabs.ai/blog/what-is-continual-learning |
| 2026-09-21 | Why Fine-Tuning Fails: The Hidden Data Quality Problem | https://adaptionlabs.ai/blog/why-fine-tuning-fails |
| 2026-09-28 | How to Improve LLM Accuracy Without Retraining Your Entire Model | https://adaptionlabs.ai/blog/improve-llm-accuracy-without-retraining |
| 2026-09-28 | The True Cost of Bad Training Data (And How to Fix It) | https://adaptionlabs.ai/blog/true-cost-of-bad-training-data |
| 2026-10-01 | Invent a Dataset: Measuring Dataset Generation Abilities with Zero Seed Data | https://adaptionlabs.ai/blog/measuring-dataset-generation-abilities-with-zero-seed-data |
| 2026-10-08 | Automated Data Quality Control With Agentic Checklists | https://adaptionlabs.ai/blog/automated-data-quality-control-with-agentic-checklists |
| 2026-10-09 | What Is Adaptive AI? How Systems That Learn and Adjust Work | https://adaptionlabs.ai/blog/what-is-adaptive-ai |
| 2026-10-15 | Visualize: The Interface for Understanding Your Data | https://adaptionlabs.ai/blog/visualize |
| 2026-10-19 | The Best Way to Reduce Your Company's AI Token Spend | https://adaptionlabs.ai/blog/reduce-company-ai-token-spend |
| 2026-10-20 | The Best Companies for Fine-Tuning AI Models in 2026 | https://adaptionlabs.ai/blog/best-companies-for-fine-tuning-ai-models |
| 2026-10-22 | 8 Tools for LLM Fine-Tuning in 2026 | https://adaptionlabs.ai/blog/llm-fine-tuning-tools |
| 2026-10-22 | Top Platforms for Synthetic Data Generation 2026 | https://adaptionlabs.ai/blog/synthetic-data-generation-platforms |
| 2026-10-22 | What Is Synthetic Data? Types, Methods, and Use Cases for AI | https://adaptionlabs.ai/blog/what-is-synthetic-data |
| 2026-10-23 | LLM Token Cost Optimization in 2026: Which Services Help? | https://adaptionlabs.ai/blog/llm-token-cost-optimization |
| 2026-10-23 | Synthetic Data vs Real Data: Differences and How to Choose | https://adaptionlabs.ai/blog/synthetic-data-vs-real-data |
| 2026-10-28 | 7 Best Platforms to Train an AI Model Easily (2026) | https://adaptionlabs.ai/blog/best-platforms-to-train-ai-model-easily |
| 2026-10-28 | 7 Continuous Learning AI Platforms Compared | https://adaptionlabs.ai/blog/continuous-learning-ai-platforms |
| 2026-10-29 | Best AI Training Data Platforms for Custom Model Development | https://adaptionlabs.ai/blog/best-ai-training-data-platforms |
| 2026-10-29 | Self-Improving Intelligence for Long-Context Medical Reasoning | https://adaptionlabs.ai/blog/self-improving-intelligence-for-long-context-medical-reasoning |
| 2026-11-04 | Most Automated Platforms for Training Custom AI Models (2026) | https://adaptionlabs.ai/blog/automated-platforms-to-train-custom-ai-models |
| 2026-11-05 | Best Platforms to Own Your AI Model (2026) | https://adaptionlabs.ai/blog/best-platforms-to-own-your-ai-model |
| 2026-11-06 | Best Platforms to Train a Custom Legal AI Model (2026) | https://adaptionlabs.ai/blog/platforms-to-train-custom-legal-ai-model |
| 2026-11-09 | 7 Top Fine-Tuning Platforms Compared (2026) | https://adaptionlabs.ai/blog/fine-tuning-platforms-compared |
| 2026-11-11 | Best Platforms to Build a Custom Healthcare AI Model | https://adaptionlabs.ai/blog/best-platforms-custom-healthcare-ai-model |
| 2026-11-12 | Expand: Automatic Data Coverage for Customized Model Training | https://adaptionlabs.ai/blog/expand |

## 8. Known gaps (no public source found; do not let the bot guess)
See README.md "Gaps" section.
