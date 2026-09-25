# SCOTUS Prediction Pipeline

Pre-registered predictions of U.S. Supreme Court case outcomes using LLM-based multi-agent deliberation.

## Overview

This repository contains:
- **Prediction code**: An agentic pipeline that simulates Supreme Court deliberation using LLMs
- **Pre-registered predictions**: Case-outcome and justice-level vote predictions for SCOTUS Terms 2022–2026, generated before opinions were handed down

## Pipeline Architecture

The prediction pipeline uses a multi-agent deliberation approach built on [LangGraph](https://github.com/langchain-ai/langgraph):

### 1. Document Ingestion
- Case materials (briefs, petitions, amicus filings) are extracted from PDFs and chunked
- Court opinions are automatically detected and excluded via content-based filtering to prevent data leakage
- Oral argument transcripts, where available, are processed separately

### 2. Hybrid Retrieval (RAG)
- BM25 (keyword) + FAISS (semantic embedding) retrieval with deduplication
- Cross-encoder reranking for syllabus generation

### 3. Case Analysis
The LLM produces a structured analysis: focal legal question, statutory provisions, both sides' positions, key precedents, and complexity rating.

### 4. Initial Voting
Each of the nine justices (modeled with individualized personas capturing judicial philosophy, voting patterns, ideological lean, and typical alliances) independently:
1. Formulates a research query based on the case analysis
2. Retrieves relevant passages from the case documents (agentic tool use)
3. Casts an initial vote with confidence score and legal reasoning

### 5. Iterative Deliberation (up to 3 rounds)
Justices see all colleagues' votes and reasoning, then decide whether to maintain or change their position. Anti-consensus-drift mechanisms prevent artificial unanimity:
- **Confidence decay**: Switching sides incurs a confidence penalty (switching cost)
- **Dissent lock**: High-confidence justices are locked after the first deliberation round
- **Vote-change cap**: At most 2 justices may flip per round to prevent cascade effects

Deliberation terminates when votes converge, the margin is stable, or the maximum number of rounds is reached.

### 6. Output
For each case, model, and replicate: initial votes, per-round vote trajectories, final votes with reasoning, predicted winner, and vote split.

## v3 Extension

`code/` also contains the v3 extension of the pipeline (per-justice opinion RAG in `scotus_v3/`,
the contamination experiment in `experiment/`, oracle fine-tuning in `oracle_experiment/`) plus
`scotus_v2/evaluate.py`. See [README_v3.md](README_v3.md) for layout, setup and the additional
release assets (justice indices, slip opinions). Note that all data paths resolve relative to
`code/`, i.e. raw PDFs go into `code/data/term_XX/`.

## Repository Structure

```
├── code/
│   ├── scotus_v2/              # Core pipeline
│   │   ├── deliberation.py     # Agentic deliberation (personas, prompts, LangGraph)
│   │   ├── retrieval.py        # Hybrid RAG (BM25 + FAISS + cross-encoder)
│   │   ├── pdf.py              # PDF extraction + opinion filtering
│   │   ├── models.py           # LLM configuration
│   │   ├── syllabus.py         # Syllabus generation
│   │   ├── config.py           # Path and configuration logic
│   │   └── keys.py             # API key management (macOS Keychain)
│   ├── config.yaml             # Model and retrieval settings
│   ├── pyproject.toml          # Python dependencies
│   └── run_term_*.py           # Per-term runner scripts
├── predictions/
│   ├── term_2022_2023/         # 58 cases × 3 models (GPT-5.2, Claude-4.6, Gemini-2.5)
│   ├── term_2023_2024/         # 59 cases × 2 models (GPT-5.2, Claude-4.6)
│   ├── term_2024_2025/         # 64 cases × 2 models (GPT-5.2, Claude-4.6)
│   ├── term_2025_2026/         # 62 cases × 2 models (committed March 18, 2026; see Pre-Registration)
│   ├── ot2025_reruns/          # September 2026 GPT-5.2 re-runs: control, transcripts, v3
│   ├── ot2026_prereg/          # October 2026 sitting forecasts (GPT-5.2)
│   ├── authorship/             # Opinion-author and separate-writing forecasts per saved run
│   └── ot2025_astra_pilot/     # GPT-6 Astra memorization pilot (one case)
├── analysis/                   # Scoring and diagnostics (see analysis/README.md)
└── syllabi/                    # Placeholder for generated syllabi
```

## Raw Case Documents

The raw input PDFs (cert petitions, briefs, amicus filings, joint appendices, oral argument transcripts) are too large to commit directly. They are attached as assets to the [`raw-data-v1` release](https://github.com/JohKrus/Scotus_Prediction/releases/tag/raw-data-v1), organized by OT term to mirror `predictions/`:

- `raw_pdfs_term_2022_2023.zip` (~1.6 GB) — 91 dockets, OT 2022 (matches `predictions/term_2022_2023/`)
- `raw_pdfs_term_2023_2024.zip` (~820 MB) — 59 dockets, OT 2023 (matches `predictions/term_2023_2024/`)
- `raw_pdfs_term_2024_2025.zip` (~715 MB) — 64 dockets, OT 2024 (matches `predictions/term_2024_2025/`)
- `raw_pdfs_term_2025_2026.zip` (~575 MB) — 62 dockets, OT 2025 (matches `predictions/term_2025_2026/`)

Each archive contains one folder per docket (`{docket}_pdfs/`) with all filings, plus a `{docket}_metadata.csv` with filing-level metadata where available. (Note: most `term_2025_2026` dockets ship without metadata CSV — those cases are pre-registered and the metadata file is generated only after the case closes.)

To use them with the pipeline, extract each into the matching `data/term_XX/` directory (the path scheme is in `code/scotus_v2/config.py`):

```bash
cd code
mkdir -p data/term_22 data/term_23 data/term_24 data/term_25
unzip raw_pdfs_term_2022_2023.zip -d data/term_22
unzip raw_pdfs_term_2023_2024.zip -d data/term_23
unzip raw_pdfs_term_2024_2025.zip -d data/term_24
unzip raw_pdfs_term_2025_2026.zip -d data/term_25
```

## Models Used

As configured in `config.yaml`:
- **OpenAI**: GPT-5.2 (snapshot `gpt-5.2-2025-12-11`)
- **Anthropic**: Claude Sonnet 4.6
- **Google**: Gemini 2.5 Flash

Each case is predicted with 2 replicates per model. Prediction files are named `{docket}_{model}_predictions.json` (e.g., `22-340_gpt52_predictions.json`, `22-340_claude46_predictions.json`).

## Pre-Registration

The predictions in `predictions/term_2025_2026/` were committed on March 18, 2026 (commit `beadb7d`); the git commit timestamps serve as the pre-registration record. That date falls inside the Term: 18 of the 58 cases decided by the end of the Term had already been decided when the forecasts were committed, so only the 40 cases decided on or after March 18, 2026 are pre-registered in the strict sense. Claude Sonnet 4.6's training data runs through January 2026, which also covers the 11 cases decided by then.

`predictions/ot2026_prereg/` holds forecasts for the October 2026 sitting (GPT-5.2, frozen v2 pipeline), made from the complete merits briefing before argument; their commit timestamp is their pre-registration record.

## CELS 2026 Revision (September 2026)

- **Scoring and diagnostics** in `analysis/` (see [analysis/README.md](analysis/README.md)), with OT2025-26 ground truth in `code/data/ground_truth/justice_votes_2025_2026.csv`.
- **Re-runs of OT2025-26 with GPT-5.2** (`code/run_ot2025_arms.py`, output in `predictions/ot2025_reruns/`): `control` (unchanged v2 pipeline; the noise floor against the March runs), `transcripts` (v2 plus the oral-argument transcript from supremecourt.gov), and `v3`. Replicates whose case analysis had silently fallen back to the generic stub in a first pass were re-run and are marked `"repaired": true`; each case's `_usage.json` records tokens and the OpenAI service tier (standard or flex, same model snapshot).
- **Authorship and separate-writing forecasts** for all saved runs (`predictions/authorship/`).
- **GPT-6 Astra pilot** for the memorization test (`predictions/ot2025_astra_pilot/`, one case, standard and anti-contamination prompts; runs at temperature 1, the only value the model accepts).
- **Pipeline changes**: `retrieval.py` caches the FAISS index per chunk list (identical results, far fewer embedding calls); `pdf.py`/`deliberation.py` detect oral-argument transcripts by content as well as by filename (`code/test_transcript_detection.py`); `scraper_dockets.py` imports `scotus_v2.config`.

## Requirements

- Python 3.12+
- API keys for OpenAI, Anthropic, and Google (environment variables `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`; macOS Keychain as fallback)
- See `code/pyproject.toml` for Python dependencies

## Authors

- Johannes Kruse
- Amit Haim
- Aniket Kesari
