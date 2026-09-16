# Scotus_Prediction — v3 extension (justice-level RAG, contamination experiment)

This extends the v2 pipeline in this repo. v3 lives in its own package and reuses the same PDF
handling, retrieval and deliberation mechanics. The only changes to v2 itself: `keys.py` now
reads API keys from environment variables (Keychain as fallback), `config.yaml` pins the GPT
snapshot, and `evaluate.py` is added.

## Layout

```
code/
  scotus_v2/           v2 pipeline + evaluate.py
  scotus_v3/           v3: per-justice opinion RAG
                         opinion_scraper.py   slip opinions from supremecourt.gov
                         opinion_parser.py    split by author into majority / concurrence / dissent
                         justice_rag.py       one FAISS index per justice, temporal filter
                         deliberation.py      v2 graph + justice_profile_node, profile-based prompts
  experiment/          contamination experiment (2x2: spoiler injection x anti-TDC prompt)
                         conditions.py, spoiler.py, anti_contamination.py, runner.py, metrics.py
  oracle_experiment/   oracle fine-tuning (data_generator.py, fine_tune.py, evaluate.py, colab_notebook.py)
  config.yaml          v2 settings + v3 sections (opinion_scraper, justice_rag)
  run_v3_setup.py      scrape opinions and build the justice indices
  run_term_*.py        per-term runners (v2, as before)
  eval_corrected.py, eval_by_model.py      evaluation by term and model
  scraper_dockets.py   docket scraper (petition, briefs, appendices -> {docket}_pdfs/ + metadata csv)
  data/
    ground_truth/      SCDB-style justice votes, OT2022-2024
    opinions/opinions_metadata.json
    oracle/            generated fine-tuning datasets
```

All paths are resolved relative to `code/` (`scotus_v3/config.py::project_dir()`), so the
data directory is `code/data/`, not a top-level `data/`.

## Large files (release assets)

- `raw_pdfs_term_*.zip` (release `raw-data-v1`) -> unzip into `code/data/term_22 ... term_25`
- `justice_indices_v1.zip` -> unzip into `code/data/justice_indices/` (14 justices, OT2019-2024)
- `opinions_pdfs_OT2019-2024.zip` -> unzip into `code/data/opinions/` (382 slip opinions)

The indices can be rebuilt from the opinions with `python run_v3_setup.py --skip-scrape`,
or from scratch (scrape + build) with `python run_v3_setup.py --terms 19,20,21,22,23,24`.
Building needs an OpenAI key for the embeddings.

## Setup

```
cd code
pip install -e .            # dependencies in pyproject.toml (plus peft/trl/torch for oracle_experiment)
```

API keys: set `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` and `GOOGLE_API_KEY` as environment
variables (`keys.py` reads those first and falls back to the macOS Keychain, which is how the
original runs were configured).

Model IDs: `config.yaml` pins `gpt-5.2-2025-12-11`, the snapshot behind the `gpt-5.2` alias used
for all runs so far. `models.py::_model_label()` maps IDs to short labels ("GPT-5.2",
"Claude-4.6") that appear in the prediction JSONs and file names; when adding a newer model,
give it a distinct label there, otherwise its output is indistinguishable from the old runs.

## Running

v2 prediction for one term: see `run_term_2025_2026.py` (model chosen by label,
`predict_case_agentic(docket, folder, llm_name=...)`, one JSON per docket).

v3 prediction: `python -c "import asyncio; from scotus_v3 import deliberation; asyncio.run(deliberation.run(terms=[24]))"`.
`run()` loads the justice indices itself; `predict_case_agentic` takes the same arguments as in
v2 plus `predict_term`, which drives the temporal filter (only opinions from terms < predict_term
are retrieved). Output: `results/predictions/agentic_v3/{docket}_predictions.json`.

Contamination experiment:
```
python -m experiment.runner --terms 24 --conditions A,B,C,D --llms GPT-5.2 --replicates 2 --name run_name
```
A = baseline, B = spoiler documents injected into the RAG corpus, C = anti-TDC prompt,
D = both. Output goes to `results/experiments/<name>/{docket}_cond{X}_{model}_rep{n}.json`.

Evaluation: `python -c "from scotus_v2 import evaluate; evaluate.run()"` (v2, mode folder in
`config.prediction_path`) / same with `scotus_v3` (reads `results/predictions/agentic_v3`),
both against `code/data/ground_truth/*.csv`.
