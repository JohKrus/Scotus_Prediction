# Task 1: input audit

Scripts: `code/revision/t1_input_audit.py` (ingestion re-run offline with the pipeline's own `pdf.is_court_opinion`, `pdf.extract_chunks` and transcript rules) and `code/revision/t1b_case_context.py` (exact reconstruction of the case-analysis context). No API calls.

- `input_inventory.csv`: one row per docket × file (5,397 files, 276 folders). Columns: document type (pipeline classifier), retained or excluded and the rule, filing date (metadata, else the e-filing timestamp in the filename), transcript flags (March filename rule, current content rule), pages, characters, chunks.
- `input_by_term.csv`: per-Term summary.
- `leak_candidates.csv`: outcome-language hits in retained chunk text. All high-precision hits are included, plus low-precision hits near the case's own name or docket. Not labeled as leaks; for manual review. `leak_counts_by_term.csv` gives all counts.
- `case_context_chunks.csv`, `case_context_by_docket.csv`, `case_context_summary.csv`: the ten chunks of every docket's case-analysis context. They equal the BM25 top 10 whenever those are distinct, which holds for 273 of 276 folders.
- `prereg_check.csv`: first-add commit of each OT2025 prediction file vs. argument and decision dates.

Key findings:
- No Supreme Court transcripts in any Term.
- Content-detected "transcripts" are lower-court transcripts in appendices (8-12% of dockets per Term).
- The Court's certified judgments are in the OT2022 inputs for 10 OT2022 and 2 OT2023 dockets. They are not in any case context.
- 18 of 57 scored OT2025 cases were decided before the 2026-03-18 commit.

Caveat: research-query retrievals inside the deliberation runs were not logged, so whether a justice agent ever retrieved a judgment chunk cannot be determined.
