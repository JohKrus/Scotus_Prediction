# Task 3: closed-book recall probe

Script: `code/revision/recall_probe.py` (prompt and scoring fixed in `PLAN.md`, commit 28bbc2e).

- `cases.csv`: 238 real cases (SCDB caption + docket) and 200 fabricated placebos (50 per Term, seed 20261007, collision-checked against all SCDB captions and dockets).
- `raw/*.jsonl`: every response as received (GPT-5.2 876, Claude Sonnet 4.6 876, Claude Opus 5.5 214). Total cost $4.25.
- `responses_parsed.csv`, `recall_by_term.csv`: claimed recall; answer rates; winner, split and author accuracy (`dont_know` as wrong and as missing); Brier; knowledge = real − fabricated.
- `effective_cutoff.csv`: one-break step model by decision month. GPT-5.2 ~Jul 2024, Claude Sonnet 4.6 ~Jun 2025, Claude Opus 5.5 ~Jan 2026.
- `cutoff_curve.png`, `cutoff_curve_monthly.csv`: monthly winner and author accuracy with a local-linear tricube smoother.
- `recall_status_by_docket.csv`: modal status per docket × model, for Task 7.

Caveat: the prompt tells the model not to guess, so the rates measure stated knowledge. Models differ in how readily they abstain: Opus 5.5 abstains most.
