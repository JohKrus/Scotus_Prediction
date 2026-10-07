# Pre-run analysis plan for the revision probes (Tasks 3 and 4)

Written 2026-10-07, before any API call for Tasks 3-4, and committed together with the
scripts that run them (`revision/recall_probe.py`, `revision/single_pass.py`,
`revision/llm.py`). The commit hash is recorded in `results/revision/SUMMARY.md`.

**Status of the evidence.** OT2025 outcomes are public: every OT2025 case scored here was
decided before this plan was written (the last on June 30, 2026). So nothing below is
pre-registered against unknown outcomes. All new results are **exploratory**. This plan fixes
prompts, case lists, models, settings and scoring before the runs, so that the analyses cannot
be tuned to their results. It does not make the results confirmatory.

## Questions

- (a) **Input shift.** Did oral-argument transcripts or post-decision docket material reach
  retrospective (OT2022-24) prompts but not OT2025 prompts? (Task 1, no API.)
- (b) **Architecture vs. contamination.** In a plain single pass, with the pipeline's own inputs,
  do the current models beat the deliberation pipeline on OT2024 and OT2025? (Task 4.)
- (c) **Exposure.** How much do the models recall, closed-book, by Term, and what is each
  model's effective legal-knowledge cutoff? (Task 3.)

## Models and settings (both tasks)

| Label | Model ID | Settings |
|---|---|---|
| GPT-5.2 | `gpt-5.2-2025-12-11` (pinned snapshot; same as the pipeline) | temperature 0.3, max_completion_tokens 4096 |
| Claude-4.6 | `claude-sonnet-4-6` (no dated snapshot exists) | temperature 0.3 (sent in the request body; SDK 1.x dropped the argument), max_tokens 4096 |
| Claude-Opus-5.5 (optional, user-supplied; OT2025 only) | `claude-opus-5-5` | The API rejects sampling parameters, so temperature cannot be set. Thinking cannot be disabled, so adaptive thinking is on (provider default) with effort `medium` set explicitly. max_tokens 16000. No server-side model fallback. |

Direct provider SDK calls, one user turn, no system prompt, no tools, no web search. Two
replicates per case and model. Every response is cached to disk on arrival
(`results/revision/<task>/raw/*.jsonl`); the resolved model name from each response is logged.
Before each task: a 3-case smoke test that prints projected cost. Any task projected above
USD 100 waits for the user's confirmation.

Claude-Opus-5.5's training data run through June 2026, so they cover every OT2025 decision. It
gets its own label and runs on OT2025 only. A large jump over Claude-4.6 would show that the
pipeline's input/prompt format can express knowledge the model has.

## Task 3: closed-book recall probe

**Cases** (`results/revision/t3_recall/cases.csv`, built by `recall_probe.py build`):
- *Real arm:* the 238 scored cases the pipeline forecast: OT2022 58, OT2023 59, OT2024 64,
  OT2025 57. These are SCDB 2026_01 dockets with a petitioner/respondent winner. OT2025 excludes
  Hamm v. Smith (DIG) and the four carried-over dockets, as in the paper. Caption = SCDB
  `caseName`; Term = SCDB `term`.
- *Fabricated placebo arm:* 50 invented cases per Term (200 total), seed 20261007. Captions are
  combinations of invented surnames, states, invented companies and federal agencies, in SCDB
  caption style. Docket numbers use that Term's format (`YY-NNN[N]`, YY = Term or Term-1;
  paid or IFP ranges). No caption equals any SCDB 2026_01 `caseName` (all Terms since 1946),
  and no real caption contains both invented parties. No docket number equals any SCDB docket.
- *Natural placebo:* the real OT2025 cases, decided after GPT-5.2's cutoff (Aug 2025) and,
  except for 11 early decisions, after Claude Sonnet 4.6's training data (Jan 2026).

**Models:** GPT-5.2 and Claude-4.6 on all 438 cases; Claude-Opus-5.5 on the 107 OT2025 cases
(57 real, 50 fabricated). 2 replicates each.

**Prompt (final text):**

```
Answer from your own knowledge only. Do not guess unless a field asks you to.
Case: {caption}, No. {docket}, Supreme Court of the United States, October Term {year}.
Return only this JSON:
{
  "recall_case": "yes" | "no",
  "decided_on_merits": "yes" | "no" | "dont_know",
  "prevailing_party": "petitioner" | "respondent" | "dont_know",
  "vote_split": "<e.g. 6-3>" | "dont_know",
  "majority_author": "<surname>" | "dont_know",
  "p_petitioner_wins": <number between 0 and 1; required, give your best estimate even if guessing>
}
```

**Scoring** (`recall_probe.py score`), by model × Term × arm:
- claimed-recall rate (`recall_case` = yes);
- answer rates for winner, split and author (anything other than `dont_know`);
- on the real arm, accuracy for winner (`prevailing_party` vs SCDB `partyWinning`), vote split
  (normalized `a-b`, vs SCDB `majVotes-minVotes`) and majority author (surname match to SCDB
  `majOpinWriter`; a per curiam counts as correct only for an answer containing "per curiam").
  Each is reported two ways: `dont_know` as missing (accuracy among answers) and `dont_know` as
  wrong (accuracy over all responses);
- Brier score of `p_petitioner_wins` (values above 1 read as percentages) against the actual
  winner; reference 0.25 for a constant 0.5;
- **knowledge** = real-arm rate minus fabricated-arm rate, for claimed recall and the three
  answer rates, same model and Term.
- Unparseable responses count as `dont_know` and are reported.
- Intervals: case bootstrap (2,000 draws, seed 20261007), replicate-level rows.

**Effective-cutoff curve:** for each model, winner and author accuracy (`dont_know` = wrong) by
decision month (SCDB `dateDecision`), monthly bins plus a LOWESS smoother (frac 0.35), with the
published cutoffs marked. Saved as PNG and CSV. **Estimated effective cutoff:** a one-break
step model for winner-correct-with-recall (`dont_know` = wrong). P = a for decisions up to
month τ and b afterwards. τ is chosen over the monthly grid to maximize the Bernoulli
likelihood; its 95% interval comes from a case bootstrap (500 draws, because the grid search is slow; at least 10 responses on each side of the break). Reported only when b < a.

## Task 4: single pass vs. the pipeline

**Cases:** the same 238 scored dockets, all four Terms; Claude-Opus-5.5 on the 57 OT2025
dockets only.

**Inputs:** identical to the pipeline's. Inputs are the docket's PDFs from the `raw-data-v1`
release, opinions excluded by `pdf.is_court_opinion`, and chunks from `pdf.extract_chunks`. Embedding and
retrieval use `retrieval.hybrid_retrieve` unchanged. The transcript split reproduces the March 2026
pipeline (filename rule), so the transcript section is empty exactly when it was empty for the
pipeline. `--no-transcripts` drops content-detected transcripts too. It is run on OT2022-24 only if
Task 1 finds transcripts reached retrospective prompts.

**Arm `neutral` (the memo's specification):**
1. Case context: `hybrid_retrieve(docs, "Supreme Court case legal arguments {docket}")`, top 10.
   This is the pipeline's case context. The pipeline does not retrieve "top-k per document type"
   in prediction mode; that setting belongs to syllabus generation.
2. Case analysis: the pipeline's prompt (paper App. C.1), verbatim. It is checked
   programmatically against `deliberation.analyze_case_node`.
3. Retrieval: the analysis's `focal_point`, `petitioner_position` and `respondent_position` each
   go through the pipeline's research tool (`_research_precedent`: hybrid top 5). Each block is
   truncated to 3,000 characters as in the pipeline's vote prompt.
4. Prediction prompt (final text):

```
You are an expert on the Supreme Court of the United States. Predict how the Court will decide this case.

CASE ANALYSIS:
{case_analysis}

LEGAL QUESTION: {focal_point}
PETITIONER'S POSITION: {petitioner_position}
RESPONDENT'S POSITION: {respondent_position}

EXCERPTS FROM THE CASE FILINGS:
{research_results}

{transcript_section}

Respond in this EXACT JSON format:
{
  "predicted_winner": "Petitioner" or "Respondent",
  "p_petitioner_wins": 0.0 to 1.0,
  "predicted_split": "Majority-minority vote count, e.g. 6-3",
  "reasoning": "Your reasoning in at most 4 sentences"
}
```

**Arm `v1` (the earlier single-pass study's prompts, found in `v1/part2_prediction.py` of the
working-paper folder):** the memo asks to use the earlier study's prompt if found. That study
was not persona-free: it asked nine persona-prompted justices to vote once each, from a case
analysis, with no research step and no deliberation. Because this differs from the memo's
"no persona" specification, both arms are run. `v1` uses its case-analysis prompt verbatim over
the same top-10 context (all documents), then the nine `JUSTICE_VOTING_PROMPT` calls verbatim
with the personas' philosophy and key factors. The prediction is the majority of nine, the
split is the tally, and p(petitioner) = petitioner votes / 9. A vote that fails to parse takes
v1's own default (lean > 0.5 → Petitioner). Both prompts are checked programmatically against
the v1 source.

**Scoring** (`single_pass.py score`), by inputs × arm × model × Term:
- replicate-level accuracy with case-bootstrap 95% intervals (2,000 draws, seed 20261007);
- always-petitioner baseline on the same cases;
- pipeline first-round tally and final tally, same model and dockets (Claude-Opus-5.5 is
  compared with the Claude-4.6 pipeline);
- paired differences (single pass − pipeline final; − first round; − baseline). Per-docket mean
  correctness is averaged over replicates, then differenced and bootstrapped over dockets;
- **McNemar test** (exact binomial) on docket-level consensus predictions. Consensus ties go to
  Petitioner, as in `analysis/score_predictions.modal_winner`;
- Brier score of p(petitioner). The pipeline's p is petitioner votes / 9 in the final round, so
  the comparison uses the same vote-share proxy;
- predicted-petitioner rate; counts of analysis fallbacks and unparsed winners. An unparsed
  winner counts as wrong.

**Primary comparisons:** OT2024 and OT2025, each model, `neutral` arm, single pass − pipeline
final. Secondary: `v1` arm; OT2022-23; Claude-Opus-5.5 on OT2025.

## Planned sensitivity analyses (Task 7)

Recomputed for the OT2025 headline (pipeline, and the single pass where applicable):
1. **Claude window exclusion:** drop cases decided before 2026-02-01 (Claude Sonnet 4.6 training
   data through January 2026), Claude only.
2. **Pre-registration-check exclusions:** drop every docket decided before its forecast file
   was first committed (`results/revision/t1_input_audit/prereg_check.csv`; first-add commit
   `beadb7d`, 2026-03-18).
3. **Recall-based exclusions:** drop cases the same model recalled correctly in Task 3. A case
   counts as recalled correctly when the modal `prevailing_party` over the two replicates is
   correct and not `dont_know`.

Within-Term mediation (Task 7): pipeline_correct ~ recall_correct + Term FE + issue group, logit,
docket-clustered SEs; the same for the single pass.

## What will not change after this commit

Prompts, case lists, models and settings, scoring rules and the exclusion rules above. Any
later deviation (for example, a model that stops being served) is reported as a deviation in
`SUMMARY.md`.
