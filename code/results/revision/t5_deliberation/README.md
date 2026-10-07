# Task 5: deliberation mechanics and institutional benchmarks

Script: `code/revision/t5_deliberation.py` (no API). Groups: model × {OT2022-24 pooled, OT2025}; by-Term versions in `deliberation_by_term.csv`. Vote-level metrics are per replicate, then averaged.

- `run_level.csv`: one row per saved run, with the reconstructed termination reason, flips per round, lock shares and switching.
- `deliberation_by_group.csv`, `deliberation_ci.csv`: safeguard binding, switching, unanimity, split distributions, and the Chief Justice in the majority.
- `agreement_*.csv`, `agreement_summary.csv`, `agreement_heatmaps.png`: 9×9 pairwise agreement, predicted (final votes) vs. actual (SCDB).

Caveats:
- Rejected flips are not logged, so whether the cap of two flips per round bound cannot be observed. The share of runs with a round of exactly two accepted flips is reported as an upper bound.
- Agents also lock when they answer `is_final: true`, or when their response fails to parse. This is not described in the paper.
