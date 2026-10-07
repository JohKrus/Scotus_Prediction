# Task 7: recall × prediction, within-Term mediation, OT2025 sensitivity

Script: `code/revision/t7_mediation.py` (system Python; no API). Inputs: Task 3 `recall_status_by_docket.csv`, the saved pipeline runs, and Task 4 `predictions_std_scored.csv`.

- `merged_runs.csv`: replicate-level rows (pipeline and single-pass arms) with the same model's closed-book recall status. Claude-Opus-5.5 single-pass runs are matched to Opus's own recall.
- `accuracy_by_recall_status.csv`: accuracy within Term by recall status (recalled_correct, recalled_wrong, dont_know, mixed = replicates disagreed).
- `mediation_logit.csv`: correct ~ recall_correct + Term FE + issue group. Logit with docket-clustered SEs; average marginal effect of recall_correct in percentage points.
- `ot2025_sensitivity.csv`: OT2025 accuracy and difference from the always-petitioner baseline under the planned exclusions. The exclusions are: decided before 2026-02-01 (applies to Claude 4.6; GPT shown for reference); decided before the 2026-03-18 forecast commit; recalled correctly by the same model; and all three.

Caveat: recall is measured once per model (2 replicates), so "recall_correct" is noisy for cases where the replicates disagree ("mixed").
