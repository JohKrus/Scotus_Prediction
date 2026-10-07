# Task 0: orientation and reproduction

Script: `code/revision/t0_reproduce.py` (no API except the model ping).

- `table5_reproduction.csv`: case accuracy by Term × model, replicate level, with case-bootstrap 95% CIs (2,000 draws, seed 20261007) and docket-consensus accuracy. All ten cells match the paper; Gemini OT2022 is 71.6 vs. 71.5, a rounding difference.
- `baseline_reproduction.csv`: always-petitioner baseline on the forecast cases (62.1, 72.9, 70.3, 66.7). Matches.
- `table8_reproduction.csv`: first-round, final and winner changed match. Flips per case match the net count (final ≠ first-round vote) in 9 of 10 cells (Claude OT2024 0.281 vs. 0.27). The stored `total_vote_changes` (gross count) differs for Gemini.
- `table10_vs_scdb.csv`: the paper's Table 10 parsed against SCDB 2026_01. 0 of 57 mismatches in winner or split.
- `reproduction_notes.txt`: consensus counts (33/33 with ties to the first replicate; 34/36 with ties to Petitioner). SCDB 2025_01 vs. 2026_01 winner codes for OT2022-24: 0 differences.
- `model_ping.json`: all pinned models served on 2026-10-07.

Ground truth for every Term: SCDB 2026_01, docket-organized files in `code/data/scdb/` (not committed). OT2025 excludes 24-872 (DIG) and the four dockets carried over to OT2026. Justice votes use the in-majority XOR petitioner-won proxy, as in `analysis/score_predictions.py`.
