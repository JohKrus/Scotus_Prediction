# Analysis scripts (CELS 2026 revision)

Scoring and diagnostics for the pipeline outputs in `../predictions/`. Unless noted,
the scripts only read saved predictions and make no API calls.

## Setup

- Python 3.11+ with `pandas`, `numpy`, `openpyxl` (and `anthropic` and `openai` for `authorship_predict.py`).
- Download the Supreme Court Database 2025 release 01 (case-centered and
  justice-centered, *Citation* versions) from <http://scdb.wustl.edu> into `data/`:
  `data/SCDB_2025_01_caseCentered_Citation.csv`, `data/SCDB_2025_01_justiceCentered_Citation.csv`.
  They are not committed (third-party data).
- `paths.py` resolves all locations, so the same scripts also run from the
  working-paper folder layout (`v2/code/`, `v2/predictions/`).

Run from this directory, e.g. `python score_predictions.py`.

## Ground truth

| File | Content |
|---|---|
| SCDB (downloaded) | OT2022-2024 case winners and justice votes |
| `data/scotus_ot2025_cases.xlsx` | OT2025-26 results (SCOTUSblog-derived; four cases hand-corrected against the opinions: 24-820, 25-95, 24-856, 25-365) |
| `parse_justice_votes_ot2526.py` | Parses the opinion attribution lines into justice votes (`load_justice_truth`) and into authorship / separate writings (`load_writings_truth`) |
| `export_ground_truth_ot2526.py` | Writes `../code/data/ground_truth/justice_votes_2025_2026.csv` in the format `scotus_v2/evaluate.py` reads |
| `ot2526_outcomes.csv` | OT2025-26 winners plus whether the FantasySCOTUS crowd called each case (built by `parse_fantasyscotus.py`) |
| `cache/scdb_truth.json` | SCDB-derived case winners used by the diagnostics |

Coding rules for OT2025-26: "would deny the petition" counts as a dissent; a
dismissal as improvidently granted is a respondent win; "concurring in the
judgment" agrees with the outcome even when "dissenting in part" follows, while
"concurring in part and dissenting in part" counts as a dissent. The four dockets
25-170, 25-459, 25-498 and 25-579 were carried over to OT2026.

## Scripts

| Script | What it does |
|---|---|
| `score_predictions.py` | Case- and justice-level accuracy, deliberation effect and calibration by Term and model (OT2022-24, SCDB) |
| `score_ot2526_partial.py` | OT2025-26 case accuracy, majority-class baseline and FantasySCOTUS crowd |
| `diagnose_ot2526.py`, `optimize_ot2526.py`, `optimize_rules.py`, `signal_test.py` | Why OT2025-26 fell below baseline: persona signal, aggregation rules fit on earlier Terms, AUC and forward-chained evaluation |
| `term_diagnostics.py`, `persona_ablation.py` | Term composition checks and persona-prior diagnostics |
| `dissent_prediction.py` | Who dissents: classification, AUC and top-k overlap against each justice's prior-Term dissent rate, case-bootstrap intervals |
| `authorship_predict.py` | One post-hoc call per saved run forecasting the majority-opinion author and separate writings (API; output in `../predictions/authorship/`) |
| `authorship_score.py` | Scores those forecasts against uniform-over-majority, sitting-balance and prior-rate baselines |
| `score_ot2025_arms.py` | The September 2026 GPT-5.2 re-runs: noise floor vs. the March runs, transcripts vs. control, v3 vs. control |
| `make_figures.py`, `make_ot2526_table.py` | Paper figures and the OT2025-26 forecast table (write to `../paper/`) |

Docket-level consensus across replicates breaks ties toward the petitioner
(`score_predictions.modal_winner`); earlier versions broke ties by Python's
randomized set order.
