# Task 4: single pass vs. the deliberation pipeline

Script: `code/revision/single_pass.py` (prompts fixed in `PLAN.md`, commit 28bbc2e). Inputs: the pipeline's own chunks (Task 1 cache), opinion filter and retrieval (`hybrid_retrieve`, via the exact `pipe_retrieve` shortcut). The transcript split follows the March 2026 code, so no transcript section, as in the pipeline.

Arms:
- `neutral`: the pipeline's case-analysis prompt, then one persona-free prediction prompt. Run on OT2022-25 for GPT-5.2 and Claude Sonnet 4.6, and on OT2025 for Claude Opus 5.5.
- `v1`: the earlier study's case-analysis prompt plus nine persona votes. Run on OT2024-25 only (cost).

Two replicates per case.

- `single_pass_by_term.csv`: accuracy with case-bootstrap CIs; baseline; pipeline first-round and final; paired differences; McNemar on docket consensus; Brier; petitioner-prediction rate.
- `predictions_std.jsonl`, `predictions_std_scored.csv`: one row per replicate-run.
- `raw/*.jsonl`: every API response, keyed by arm, docket, replicate and step.
- `contexts_std.jsonl`: case-context and v1-context chunk sources per docket.
- `retrieval_equivalence_check.txt`: `pipe_retrieve` vs. full `hybrid_retrieve` (12/12 identical).

Headline: on OT2024 the neutral single pass beats the pipeline (Claude +10.2 pts [3.1, 18.8]; GPT-5.2 +7.8 [0.8, 14.8]). On OT2025 both models are about equal to the pipeline and below the baseline. The v1 persona-vote prompt matches the pipeline. Opus 5.5 reaches 89.5% on OT2025, but its training data cover the Term.

Cost: about $70 at list prices (GPT-5.2 on the flex tier).
