# Task 6: memorization fingerprints, decomposition, difficulty anchors

Script: `code/revision/t6_fingerprints.py` (no API). Amicus counts: Task 1 inventory (amicus filings per docket; cover-page check where filing metadata are missing, mostly OT2025).

- `fingerprints_by_term.csv`: exact split, dissenter Jaccard, cross-bloc vs. bloc-consistent vote accuracy, replicate agreement, mean confidence, gap to baseline and to FantasySCOTUS (not case-matched).
- `kitagawa.csv`: change from pooled OT2022-23 to OT2024 and to OT2025, split into composition and within-group parts with bootstrap CIs. Groups: `issue_group_mapping.csv`, counts in `issue_group_counts.csv` (Judicial Power kept separate though it has fewer than 8 cases in OT2022-23).
- `pooled_logit.txt`, `pooled_logit_coefs.csv`: correct ~ Term×model cells + issue group + lower-court direction + state source + log(1 + amicus), replicate rows, docket-clustered SEs.
- `prominence_logit.csv`, `prominence_binned.csv`, `prominence_dose_response.png`, `amicus_vs_division.csv`: the prominence dose-response.

Finding: composition explains almost none of the OT2024 or OT2025 drop; nearly all of it is within issue groups.
