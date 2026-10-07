# Prediction JSON schema (pre-registered and retrospective runs)

Files: `predictions/term_{2022_2023,2023_2024,2024_2025,2025_2026}/{docket}_{model}_predictions.json`
(model tag `gpt52`, `claude46`, `gemini25`). Each file is a JSON **list** with one object per
replicate (2 per docket and model; 1,154 objects in all). Written by
`scotus_v2/deliberation.py::predict_case_agentic` (lines 781-809).

| Field | Type | Content |
|---|---|---|
| `docket` | str | Docket number as on supremecourt.gov |
| `llm_model` | str | Label from `models._model_label()`: `GPT-5.2`, `Claude-4.6`, `Gemini-2.5`. No resolved model ID or snapshot is logged. |
| `mode` | str | Always `agentic_v2` |
| `replicate` | int | 1 or 2 |
| `has_transcripts` | bool | Whether the transcript section was non-empty. **False in all 1,154 runs.** |
| `case_analysis` | dict | Parsed JSON of the case-analysis call: `focal_point`, `legal_provision`, `petitioner_position`, `respondent_position`, `key_precedents`, `legal_complexity`. See "Silent fallbacks" below. |
| `initial_votes` | dict[justice → vote] | Round-1 votes (after each justice's research query and retrieval): `vote` (`Petitioner`/`Respondent`), `confidence` (0-1, as stated by the model), `reasoning` (text), `is_final` (false) |
| `votes_per_round` | list[dict[justice → {vote, confidence, is_final}]] | Element 0 = initial votes; elements 1..k = votes after each deliberation round. **No per-round reasoning** (only the final round's reasoning survives, in `justice_votes`). `is_final` = locked (confidence ≥ 0.85 from round 2 on), or forced final after a parse/API error. |
| `justice_votes` | dict[justice → vote] | Final-round votes with `vote`, `confidence`, `reasoning`, `is_final` |
| `petitioner_votes`, `respondent_votes` | int | Final tally |
| `predicted_winner`, `vote_split` | str | Final winner and split (e.g. `6-3`) |
| `initial_winner`, `initial_vote_split` | str | Same for round 1 |
| `average_confidence` | float | Mean final confidence over the nine justices |
| `deliberation_rounds` | int | Number of deliberation rounds executed (2 or 3; `len(votes_per_round) - 1`) |
| `total_vote_changes` | int | Gross count of round-to-round vote changes, summed over rounds |

## What is and is not logged

- **Termination reason: not logged.** It can be reconstructed from `votes_per_round` and the
  routing rule in `should_continue_deliberating()` (at least two rounds; a third round when the
  margin is ≤ 3 or votes changed in round 2; stop after round 3). In the data, 809 runs stopped
  after two rounds and 345 after three.
- **Retrieved chunks and prompt contexts: not logged.** Neither the case context (10 chunks from
  `hybrid_retrieve`), the per-justice research passages, nor the justices' research queries are
  saved. Inputs can only be reconstructed by re-running the deterministic ingestion offline
  (Task 1).
- **Transcript text: not logged**, and `has_transcripts` is false everywhere. Under the March 2026
  code (commit `c53f23f`), `_split_docs_by_type` recognised transcripts by filename only
  (`transcript|oralarg`). A transcript with an opaque filename would not be split out and would
  instead enter `case_context` and the research retrieval as an ordinary document. Whether that
  happened is a Task 1 question.
- **Attempted-but-capped vote changes: not logged.** The cap rejects changes in
  `deliberation_node` (lines 615-625) and writes only a log line; the rejected proposal is
  discarded and the justice keeps the previous vote with `is_final = False`. So whether the cap
  of two flips per round ever bound cannot be determined from the saved files.
- **Deliberation reasoning per round: not logged** (only the last round's reasoning).
- **Model snapshot / resolved model ID: not logged.** `config.yaml` now pins
  `gpt-5.2-2025-12-11`; the March runs used the `gpt-5.2` alias (which resolved to that
  snapshot, per the coauthor) and `claude-sonnet-4-6` (no dated snapshot exists).
- **Timestamps: not logged** in the files; only git commit dates exist.

## Silent fallbacks found in the saved runs

`analyze_case_node` falls back to a stub analysis (`focal_point` = "Legal interpretation
question", positions "Position unclear") when the call fails or its JSON does not parse; the
parser (`_parse_json`) takes the first flat `{...}` in the response, so a response with nested
objects can also yield a wrong partial dict. Counts:

| Folder | Model | Stub analysis | Malformed analysis (no `focal_point`) |
|---|---|---|---|
| term_2022_2023 | Claude-4.6 | 2 | 1 |
| term_2022_2023 | Gemini-2.5 | 2 | 0 |
| term_2023_2024 | Claude-4.6 | 6 | 0 |
| term_2024_2025 | Claude-4.6 | 5 | 2 |
| term_2025_2026 | Claude-4.6 | 5 | 0 |
| all | GPT-5.2 | 0 | 0 |

`initial_votes_node` falls back to a lean-based default vote (conservative lean > 0.5 →
Petitioner) with confidence 0.3 (reasoning "Fallback") or 0.2 (reasoning = error text). Gemini-2.5
has 24 such votes in 17 runs; GPT-5.2 has 1 (OT2024); Claude none. These runs are kept in all
analyses (as in the paper); Task 5 reports them.
