# Revision diagnostics: summary (2026-10-07)

Work follows `revision_memo_claude_code.md`. Code is in `code/revision/` and outputs are in
`code/results/revision/<task>/`. Nothing under `predictions/` was modified, and no paper numbers
were changed. All new results are **exploratory**: OT2025 outcomes were public before any of
this was run.

**Analysis plan commit (Task 2):** `28bbc2e1eca80103ec90e59528b3a6e6491b18d9`, 2026-10-07
17:15:46 +03:00, "Pre-run analysis plan for revision probes". It was pushed to
`cels-2026-revision` before any Task 3-4 API call. Deviations from the plan are listed at the end.

---

## Direct answers

**(a) Is input shift real?** Not for transcripts or input volume. The OT2022 inputs do contain
a different leak.
- No Supreme Court oral-argument transcript is in any Term's inputs. The transcript section of
  the prompts was empty in all 1,154 saved runs. The content detector flags 8-12% of dockets in
  every Term, but every flagged file is a joint or petition appendix that reproduces a
  lower-court hearing transcript.
- Input volume is similar across Terms: median chunks per docket are 1,095, 1,025, 975 and
  1,023 for OT2022-25, and median amicus filings are 9, 9, 7 and 7.
- **Retrospective-only leak:** the OT2022 input folder contains the Court's own certified
  judgment documents for 10 of the 58 OT2022 cases (for example
  `22-138_Judg_Vac_Rem_MANDATE_COSTS_...pdf`: "the judgment of the above court is vacated with
  costs"), plus two OT2023 cases (22-193 and 22-340, used only in the Gemini runs). There is
  also a post-decision dismissal stipulation in 22O145. The opinion filter does not catch these
  files. They did **not** enter any case-analysis context (reconstructed exactly; see Task 1).
  They could have entered a justice's research retrieval, which is not logged. As a proxy,
  the key precedents in the saved case analyses were used as research queries: 6 of 291
  top-5 retrievals hit a judgment chunk, in 3 of the 10 dockets (`research_exposure_proxy.csv`).
  So exposure was possible but uncommon.
  - Pipeline accuracy on those 10 dockets is 90.0% (Claude) and 100% (GPT-5.2), against 80.2%
    and 86.5% on the other 48.
  - The same 10 cases are also better recalled closed-book: 95% and 90% vs. 79% and 72%. So
    the gap may reflect memorization rather than the leak.

**(b) Does a plain single pass beat the pipeline on OT2024 and OT2025?** → *filled in below
after the Task 4 run (see Task 4).*

**(c) What do the models know, and what are their effective cutoffs?** Closed-book winner
recall, with `dont_know` counted as wrong:

| | OT2022 | OT2023 | OT2024 | OT2025 |
|---|---|---|---|---|
| Claude Sonnet 4.6 | 81.9% [71.6, 91.4] | 86.4% [78.0, 94.1] | 60.9% [49.2, 73.4] | 1.8% [0, 5.3] |
| GPT-5.2 | 75.0% [63.8, 85.3] | 44.9% [33.1, 57.6] | 0.0% | 0.0% |
| Claude Opus 5.5 | – | – | – | 16.7% [7.0, 27.2] |

On the 200 fabricated placebo cases, no model claimed to recall a case or gave a winner,
split or author. Estimated effective cutoffs (one-break step model; τ = last decision month
before the break; 95% case-bootstrap interval):
- **GPT-5.2: about July 2024** [June-July 2024], against a published cutoff of August 2025.
- **Claude Sonnet 4.6: about June 2025** [May-June 2025], against August 2025 (reliable
  knowledge) and January 2026 (training data).
- **Claude Opus 5.5: about January 2026** [January-February 2026], against June 2026.

Two consequences for the paper. **GPT-5.2 knows nothing about OT2024**, so for GPT-5.2 the
OT2024 Term is already clean. **Claude Sonnet 4.6 knows most of OT2024**, so for Claude the
OT2024 drop is *not* a loss of knowledge (see Task 7).

---

## Task 0: orientation and reproduction (`t0_reproduce/`)

- **Table 5 reproduces exactly** for all ten Term × model cells. Gemini OT2022 is 71.6 here vs
  71.5 in the paper; 83/116 = 71.55, so the gap is rounding. The baselines reproduce
  (62.1, 72.9, 70.3, 66.7). Bootstrap intervals differ by a point or so from the paper's
  (different seed).
- **Table 8:** first-round, final and winner-changed reproduce. "Flips/case" matches the
  *net* count (justices whose final vote differs from their first-round vote) in 9 of 10 cells.
  Claude OT2024 is 0.281 vs 0.27 in the paper. It is not the stored `total_vote_changes` (gross
  round-to-round changes), which gives 0.37 and 0.38 for Gemini. The table caption should say
  which count is reported.
- **Table 10 "Actual" column = SCDB 2026_01** for all 57 scored cases (winner and split).
  Consensus counts are 33/57 for both models with ties going to the first replicate (as the
  caption says). With ties going to Petitioner they are 34 (Claude) and 36 (GPT-5.2). Seven
  Claude dockets and ten GPT dockets have split replicates. SCDB 2025_01 and 2026_01 agree on
  every OT2022-24 winner.
- OT2025 ground truth is SCDB 2026_01 (`common.truth_table`), excluding Hamm (DIG) and the four
  carried-over dockets, as in the paper.
- **Models still served (pinged 2026-10-07):** `gpt-5.2-2025-12-11`, `claude-sonnet-4-6`,
  `claude-opus-5-5`.
- **Prediction JSON schema:** `code/revision/SCHEMA.md`. Logged: initial, per-round and final
  votes with confidence; final-round reasoning only. **Not logged:** retrieved chunks or
  prompts, research queries, termination reason (reconstructable), transcript text (none
  existed), attempted-but-capped flips, and the resolved model ID. Silent fallbacks: 18 Claude
  runs and 2 Gemini runs used the stub case analysis ("Legal interpretation question"), and 3
  Claude analyses parsed only partially. Gemini has 24 fallback initial votes in 17 runs.
- **Earlier single-pass study:** `v1/` in the working-paper folder (not in the repo). It
  contains the code and prompts (`part2_prediction.py`: one case analysis, then nine *persona*
  votes, no research and no deliberation) and OT2023 progress files. It has **no OT2022 or
  OT2024 predictions and no scoring script** behind Table 2 or Figure 6.
- **Existing contamination experiment (`experiment/`, `oracle_experiment/`):** code only.
  `results/experiments/` exists neither in the repo nor locally. `data/oracle/*.jsonl` holds
  generated fine-tuning sets (206 items each), but no training or evaluation outputs. Per the
  coauthor's notes, only a very small pilot was run. Nothing to summarize quantitatively.
- **Environment problems found on this machine (they matter for anyone re-running here):**
  1. Windows Application Control now blocks faiss-cpu's DLL. It loaded in September.
     `scotus_v2.retrieval` then silently falls back to BM25-only retrieval and logs only
     "FAISS batch failed". For Task 4, `revision/faiss_shim.py` supplies an exact numpy
     IndexFlatL2.
  2. Parts of statsmodels are blocked as well (analysis scripts use the system Python).
  3. The anthropic SDK 1.x dropped `temperature`. Sonnet 4.6 still accepts it in the request
     body; Opus 5.5 rejects sampling parameters.

## Task 1: input audit (`t1_input_audit/`)

Files: `input_inventory.csv` (5,397 files in 276 docket folders), `input_by_term.csv`,
`leak_candidates.csv`, `leak_counts_by_term.csv`, `case_context_*.csv`, `prereg_check.csv`.

- **Transcripts:** see (a). Under the March 2026 code (filename rule), the transcript channel
  was empty for every docket. **The current HEAD routes these appendices into the
  "oral argument" channel:** the content rule added in September flags them. That affected the
  September OT2025 re-runs (control, transcripts and v3 arms) for 5 dockets: 24-109, 24-813,
  25-197, 25-51 and 25-95. The detector needs a Supreme-Court-specific header test before it is
  used again.
- **The "hybrid" retrieval is BM25-only in prediction mode.** `hybrid_retrieve` lists the BM25
  top k first, removes repeated texts and stops at k. The FAISS results therefore enter only
  when BM25 returns repeated chunk texts. That happens in 1 of 91, 0, 0 and 2 of 62 dockets'
  case contexts, and in 0 of the Task 4 research calls (see Task 4). The paper's description
  of semantic retrieval does not match what the prompts received. A side effect: the case
  context is deterministic and identical across models and replicates.
- **Case contexts reconstructed** (`case_context_chunks.csv`): no post-decision chunk is in
  any case context. One post-argument chunk appears (OT2024 23-1067: a letter citing a
  lower-court decision), and one in OT2025 (Callais supplemental brief, filed after the
  reargument order).
- **Opinions present on disk and correctly excluded:** OT2022 13 files, OT2024 2, OT2025 4.
  The OT2025 four are 24-1159, 25-52, 25-180 and 25-51; the OT2025 folders were scraped after
  those decisions.
- **Leak scan** (manual-review candidates; counts by Term in `leak_counts_by_term.csv`):
  high-precision outcome phrases near the case's own name or docket appear 47, 6, 19 and 15
  times (OT2022-25; OT2022 includes 11 hits in the 12 Supreme Court judgment files, found
  by the added pattern "ordered and adjudged by this Court that the judgment"). Apart from
  those judgment files, the sampled hits are
  lower-court judgments in petition appendices (for example, "the judgment of the district
  court is AFFIRMED"). No slip-opinion marker ("NOTICE: This opinion is subject to formal
  revision", "delivered the opinion", "Cite as") appears with an own-case mention in retained
  text.
- **Pre-registration check (`prereg_check.csv`):** all 124 OT2025 prediction files were first
  added in a single commit, `beadb7d`, on 2026-03-18 08:46 +01:00. Of the 57 scored dockets:
  - **18 were decided before the commit** (earliest 2025-11-24);
  - 42 had been argued before it (only 15 were committed before argument);
  - 11 were decided before 2026-02-01;
  - **3 were decided before the GPT-5.2 snapshot date** (2025-12-11), so those forecasts
    necessarily postdate the decision.

  The paper's "generated before any opinion issued" is false for 18 of 57 cases.
- **OT2024 reconciliation (Table 2: GPT-4o 76.6% vs. Figure 6: about 81%):** the materials are
  not available. **Request from the authors (Johannes):** the earlier study's OT2022 and
  OT2024 prediction files (per model and replicate); the OT2024 case list; whether inputs
  included transcripts; and the script that produced Table 2 and Figure 6, including the
  scoring rule (majority over three replicates vs. replicate-level) and how vacaturs and DIGs
  were coded.

## Task 3: closed-book recall probe (`t3_recall/`)

1,966 calls in all: GPT-5.2 876 and Claude Sonnet 4.6 876 (real and fabricated arms, OT2022-25),
and Claude Opus 5.5 214 (OT2025). Cost: $1.14, $1.66 and $1.45. Every response parsed.

| Model | Term | Claimed recall | Winner (dk = wrong) | Winner (dk = missing) | Exact split (dk = wrong) | Author (dk = wrong) | Brier |
|---|---|---|---|---|---|---|---|
| Claude 4.6 | 2022 | 100% | 81.9 | 83.3 | 88.8 | 77.6 | 0.139 |
| Claude 4.6 | 2023 | 100% | 86.4 | 91.9 | 82.2 | 82.2 | 0.088 |
| Claude 4.6 | 2024 | 72.7% | 60.9 | 84.8 | 45.3 | 28.9 | 0.179 |
| Claude 4.6 | 2025 | 2.6% | 1.8 | (1 answer) | 1.8 | 0 | 0.274 |
| GPT-5.2 | 2022 | 86.2% | 75.0 | 88.8 | 67.2 | 75.9 | 0.133 |
| GPT-5.2 | 2023 | 46.6% | 44.9 | 98.1 | 36.4 | 36.4 | 0.161 |
| GPT-5.2 | 2024 | 2.3% | 0 | – | 0 | 0 | 0.249 |
| GPT-5.2 | 2025 | 0% | 0 | – | 0 | 0 | 0.250 |
| Opus 5.5 | 2025 | 86.8% | 16.7 | 95.0 | 12.3 | 13.2 | 0.122 |

Fabricated arm: 0% claimed recall and 0% answers for winner, split and author, for every model
and Term. So knowledge (real minus fabricated) equals the real-arm rates. Opus 5.5 claims to
"recall" 87% of OT2025 cases but names a winner only for the cases decided before February
2026 (82% correct there, 1% after). Its probability forecasts stay informative after that date
(Brier 0.145, against 0.25 for a constant 0.5). Claude 4.6 named one OT2025 winner (Callais,
24-109).
Figure: `cutoff_curve.png`. Data: `cutoff_curve_monthly.csv`, `effective_cutoff.csv`. Docket
statuses for Task 7: `recall_status_by_docket.csv`.

Caveat: the probe measures what a model *says* it knows under an instruction not to guess.
GPT-5.2 abstains more readily than Claude, and Opus 5.5 abstains most. "Answered" accuracy is
92-98% wherever a model answers.

## Task 4: single pass vs. pipeline (`t4_single_pass/`)

→ *filled in after the run.*

## Task 5: deliberation mechanics (`t5_deliberation/`)

Pooled OT2022-24 vs. OT2025, by model. Vote-level metrics are computed per replicate and then
averaged. Files: `deliberation_by_group.csv`, `deliberation_by_term.csv`, `deliberation_ci.csv`,
`agreement_*`, `agreement_heatmaps.png`, `run_level.csv`.

- **Safeguard binding.**
  - Agents at confidence ≥ 0.85 after the first conference round (locked from round 2):
    Claude 53% (OT2022-24) and 37% (OT2025); GPT-5.2 5% and 2%; Gemini 94%.
  - **An undocumented lock:** the deliberation prompt asks for `is_final`, and an agent that
    answers `true` is locked for the remaining rounds. In runs that reach round 3, 71% of
    Claude agents enter it locked: 31% through the confidence rule and **40% through
    self-declared finality** (or a parse failure). For GPT-5.2 the figures are 2% and 1%. The
    paper describes only the confidence lock.
  - Rounds: 74-75% of runs stop after two rounds (OT2022-24) and 56-59% in OT2025. The rest
    run a third round, mostly because the margin was 6-3 or closer.
  - **Cap binding cannot be observed.** Rejected flips are not logged. As an upper bound, a
    round with exactly two accepted flips occurs in 2.2% (Claude), 5.5% (Gemini) and 0%
    (GPT-5.2) of runs.
- **Switching.** Net per-vote switching (final ≠ initial) is Claude 3.0% (OT2022-24) and 4.6%
  (OT2025), GPT-5.2 0.5% and 0.8%, and Gemini 3.5%. The reference figure for the real Court
  (Lax & Rader 2015) is about 7% conference-to-final. 95% of Claude's switches and 100% of
  GPT-5.2's move *toward* the initial majority (Gemini 76%). The winner changes in 0.6% and 1.8%
  of Claude runs and never for GPT-5.2.
- **Consensus.**
  - Predicted unanimity: Claude 58% (OT2022-24) and 47% (OT2025); GPT-5.2 30% and 32%. Actual:
    46% and 49%.
  - Predicted 9-0 against actual unanimity: precision and recall are 0.57/0.71 for Claude and
    0.62/0.41 for GPT-5.2 (OT2022-24).
  - Both models under-predict 6-3 decisions: 10% and 16% of predictions vs. 23% actual
    (OT2022-24).
- **Agreement structure.**
  - Correlation of the 36 off-diagonal pairwise-agreement cells, predicted vs. actual:
    Claude 0.77 (OT2022-24) and 0.68 (OT2025); GPT-5.2 0.69 and 0.83; Gemini 0.71.
  - Mean absolute difference: 0.06-0.08.
  - Claude over-predicts cross-bloc agreement: 0.77 vs. 0.68 actual.
- **Chief Justice.** The Roberts agent is in the predicted majority in 97% (Claude) and 95%
  (GPT-5.2) of runs, against 96% actually.

## Task 6: fingerprints, decomposition, difficulty anchors (`t6_fingerprints/`)

- **Fingerprints** (`fingerprints_by_term.csv`, replicate level, case-bootstrap CIs). Values
  below are OT2022 / OT2023 / OT2024 / OT2025.
  - Exact-split accuracy: Claude 45 / 42 / 32 / 40%; GPT-5.2 41 / 25 / 24 / 29%.
  - Dissenter-set Jaccard (actually non-unanimous cases): Claude 0.32 / 0.27 / 0.13 / 0.16;
    GPT-5.2 0.37 / 0.36 / 0.22 / 0.31.
  - Cross-bloc votes are predicted worse than bloc-consistent votes in OT2022-24 (36-55% vs.
    65-81%). OT2025 has only 36 cross-bloc votes.
  - Replicate agreement on the winner: 0.84-0.98.
  - **The pipeline's split accuracy is far below the closed-book probe's** (Claude 45% vs. 89%
    on OT2022). The pipeline expresses little of the memorized vote structure.
- **The pipeline's own case analyses sometimes state the outcome** (`t6b_analysis_selfleak.py`,
  `analysis_selfleak_*.csv`). Step 1 of every run, written before any justice votes, cites the
  case itself as decided. This happened in 5 of 116 GPT-5.2 runs on OT2022, for example:
  - "Helix Energy Solutions Group, Inc. v. Hewitt, 598 U.S. 39 (2023)";
  - "Slack Technologies, LLC v. Pirani, 598 U.S. ___ (2023) (Supreme Court ultimately held …
    vacated/remanded …)";
  - Glacier Northwest: "if considering the Supreme Court's eventual merits decision: held …".

  Elsewhere it occurs in ≤ 1.7% of runs (candidates for manual review). The filings in the
  context contain no opinion of the case, so this is memory surfacing inside the pipeline.
  It is direct evidence for the contamination channel.
- **Crowd gap** (not case-matched): OT2025 −37.7 (Claude) and −31.5 (GPT-5.2) points; OT2022
  +6.9 and +13.8. Against the always-petitioner baseline: OT2022 +19.8 and +26.7; OT2025 −9.6
  and −3.5.
- **Kitagawa decomposition** (`kitagawa.csv`; issue groups: Criminal Procedure; Rights = Civil
  Rights, First Amendment, Due Process, Privacy; Judicial Power, kept separate; Economic & other
  = everything else; mapping in `issue_group_mapping.csv`). Pooled OT2022-23 → OT2024: GPT-5.2
  −18.2 pts [−30.6, −6.0], of which composition −2.6 [−7.4, 1.3] and within-group −15.6;
  Claude −18.6, of which composition −1.7 and within-group −16.8. → OT2025: GPT-5.2 −21.5
  (composition −0.1); Claude −27.2 (composition +1.4). **Issue composition, Judicial Power
  included, explains almost none of the drop.** That cuts against the paper's composition
  argument.
- **Pooled logit** (`pooled_logit_coefs.csv`, replicate rows, docket-clustered SEs, Term ×
  model cells). Term × model effects stay large after ex-ante covariates: OT2024 −1.4, OT2025
  −1.7 to −1.9 logits vs. OT2022 GPT-5.2. Issue group and log(1 + amicus count) are null.
  State-court source is +1.07 (p = .03).
- **Prominence dose-response** (`prominence_logit.csv`, `prominence_dose_response.png`). No
  significant slope. In the binned data, Claude's OT2022-24 accuracy rises from 70% to 86%
  across amicus quartiles, consistent with prominent cases being better remembered. GPT-5.2 is
  flat, and its OT2025 accuracy falls with prominence (78% → 46%). Prominent cases are only
  slightly more often divided (53% → 59%).

## Task 7: mediation and sensitivity (`t7_mediation/`)

→ *final numbers after the Task 4 run; pipeline results below are final.*

- **Pipeline accuracy by recall status, within Term** (`accuracy_by_recall_status.csv`):
  - GPT-5.2 OT2022: 97.7% on recalled-correct cases vs. 62.5% on `dont_know`.
  - Claude OT2022: 87.2% on recalled-correct cases vs. 61.1% on recalled-wrong cases.
  - **Claude OT2024: 68.4% on recalled-correct cases vs. 70.6% on `dont_know`.** The OT2024
    pipeline does not use what Claude knows.
- **Within-Term logit** (pipeline_correct ~ recall_correct + Term FE + issue group, clustered):
  average marginal effect of a correct recall is +31.6 pts [8.6, 54.6] for GPT-5.2 and +13.5
  [−0.6, 27.5] for Claude.
- **OT2025 headline sensitivity** (`ot2025_sensitivity.csv`, pipeline, replicate level):

| Variant | Claude 4.6 | GPT-5.2 | Baseline |
|---|---|---|---|
| All 57 | 57.0 [44.7, 68.4] | 63.2 [51.8, 74.6] | 66.7 |
| Excl. decided before 2026-02-01 (n = 46; Claude window) | 53.3 [40.2, 66.3] | (62.0) | 63.0 |
| Excl. decided before the forecast commit (n = 39) | 53.8 [38.5, 69.2] | 61.5 [47.4, 75.6] | 66.7 |
| Excl. recalled correctly (Claude n = 56, GPT n = 57) | 58.0 | 63.2 | 67.9 / 66.7 |
| Excl. all three (Claude n = 38, GPT n = 39) | 55.3 | 61.5 | 68.4 / 66.7 |

The headline conclusion survives every exclusion: neither model beats the baseline.

## Task 8: paper fixes

The user supplied the LaTeX source (`overleaf/main.tex`), so the fixes are a **diff for
review**. It is kept with the manuscript, not in this public repository:
`revision_outputs/main_proposed.diff`, `main_proposed.tex` and `references_proposed.diff` in
the working-paper folder, generated by `revision_outputs/make_paper_diff.py`. The proposed file
compiles with no undefined references. New-analysis numbers are left as `\TODO{}`. Items:
- stale OT2025 text (Sec. 3.2 ground truth "not available"; Sec. 3.4 "future revision");
- cutoff inconsistency (Sec. 3.4 "not publicly documented"; Sec. 5.2 "should be read as
  memorization");
- transcript statements (Sec. 3.2, App. C, Sec. 6);
- pre-registration wording (abstract, intro, Sec. 3.4, Sec. 4, App. E);
- retrieval description (top-20-per-type is the syllabus pipeline; prediction is BM25 top 10
  and top 5);
- the self-declared `is_final` lock and the actual termination rule;
- SCDB release 2026_01;
- the Table 10 vs. headline sentence;
- the "flips" definition;
- three broken references (`sec:deliberation` twice, `sec:prereg`);
- two empty footnotes;
- the "[Moved from …]" notes;
- the abstract's first sentence;
- typos ("predictino", "mutli-agent", "instantitate", "long-minded", "commiting");
- the Unikowsky access date.

The GPT-4o 76.6 vs. 81 conflict is flagged in a comment (materials needed). The repo README is
already correct about OT2025 ground truth. `code/run_term_2025_2026.py`'s docstring still says
"No ground truth available"; it is left untouched, as it is the pre-registration runner.

Also for the paper:
- Sec. 3.2 says "approximately 50-200 chunks per case"; the median is about 1,000, with up to
  12,367 chunks (SFFA).
- The opinion filter claim ("no model could read the disposition") needs the Task 1
  qualification.
- The OT2025 scored sample includes unargued per curiams and one stay application, while
  Sec. 3.2 says "argued cases".

## Deviations from PLAN.md

1. **Task 4 scope, cut for cost at the user's request.** The `v1` arm was run on OT2024-25
   only, not on all four Terms. Claude Opus 5.5 ran the `neutral` arm only. GPT-5.2 calls used
   OpenAI's flex service tier (same snapshot, half price).
2. **Task 4 retrieval implementation.** `pipe_retrieve` returns the BM25 top k when it has no
   repeated texts and otherwise calls `retrieval.hybrid_retrieve`. This provably equals
   `hybrid_retrieve`; 12 of 12 checks against the full function with embeddings were identical
   (`t4_single_pass/retrieval_equivalence_check.txt`). FAISS uses an exact numpy stand-in
   because the compiled extension is blocked on this machine.
3. **Task 3 smoother.** LOWESS without robustness iterations (own implementation; the
   statsmodels extension is blocked). The step-model bootstrap uses 500 draws, as stated in the
   plan.

## Decisions needed from the authors

1. **The OT2022 judgment leak.** Re-run OT2022 for the 10 affected dockets with the judgments
   removed (about $10 with GPT-5.2 and Claude), or report it as a limitation? The deliberation
   runs do not log research retrievals, so exposure cannot be ruled out after the fact. The
   accuracy contrast (90% / 100% vs. 80% / 87%) is confounded with closed-book recall.
2. **Contamination framing.** The measured cutoffs move OT2024 to "clean" for GPT-5.2 and to
   "largely known" for Claude. The paper's "OT2024 near the edge of the cutoff / harder docket"
   discussion and its composition argument need rewriting. Kitagawa finds no composition
   effect, and Claude's OT2024 drop happens despite recall.
3. **Pre-registration language.** 18 of 57 scored cases were decided before the forecasts were
   committed, and 3 before the GPT-5.2 snapshot existed. Should the strictly pre-registered 39
   be the headline?
4. **Retrieval description.** Describe the pipeline as BM25 retrieval, or treat the
   non-functional semantic component as a known defect?
5. **Undocumented `is_final` lock.** Describe it; it is a fourth safeguard and probably part of
   why deliberation moves so little for Claude.
6. **Transcript detector at HEAD.** Fix before any further runs. It misroutes appendices into
   the oral-argument channel. Decide whether the September OT2025 re-runs (5 dockets affected)
   need repair.
7. **Materials for Table 2 / Figure 6** (see the Task 1 request list).
8. **Task 9** (no-safeguards pilot) was skipped today at the user's request.
