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
│   └── term_2025_2026/         # 62 cases × 2 models (pre-registered, no ground truth yet)
└── syllabi/                    # Placeholder for generated syllabi
```

## Models Used

As configured in `config.yaml`:
- **OpenAI**: GPT-5.2
- **Anthropic**: Claude Sonnet 4.6
- **Google**: Gemini 2.5 Flash

Each case is predicted with 2 replicates per model. Prediction files are named `{docket}_{model}_predictions.json` (e.g., `22-340_gpt52_predictions.json`, `22-340_claude46_predictions.json`).

## Pre-Registration

The predictions in `predictions/term_2025_2026/` were generated **before** the Supreme Court handed down its opinions for the October 2025 Term. The git commit timestamps serve as proof of pre-registration.

## Requirements

- Python 3.12+
- API keys for OpenAI, Anthropic, and Google (stored in macOS Keychain)
- See `code/pyproject.toml` for Python dependencies

## Authors

- Johannes Kruse
- Amit Haim
- Aniket Kesari
