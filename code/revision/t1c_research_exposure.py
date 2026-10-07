"""Task 1 (part 3): could the leaked OT2022 judgment files reach a research retrieval?

The deliberation runs did not log the justices' research queries. As a proxy,
this replays the Task 4 single-pass research retrievals for the dockets whose
inputs contain a Supreme Court judgment. It uses the three queries built from
each run's case analysis (focal point, petitioner position, respondent position),
the top 5 per query, all models and replicates. It reports how often a judgment
chunk is retrieved.

    python -m revision.t1c_research_exposure
"""
from __future__ import annotations

import json

import pandas as pd

from revision import common as C
from revision import llm, single_pass as sp

JUDGMENT_DOCKETS = ["22-105", "22-138", "22-148", "22-166", "22-174", "22-179", "22-196", "22-200",
                    "22-210", "22-227"]


def main():
    o = C.out("t1_input_audit")
    inv = pd.read_csv(o / "input_inventory.csv", low_memory=False)
    late = set(inv[(inv.term_dir == "term_22") & (inv.filed_after_decision == True)].file)
    rows = []
    for model in ("GPT-5.2", "Claude-4.6"):
        cache = llm.Cache(C.OUT_DIR / "t4_single_pass" / "raw" / f"std_neutral_{model}.jsonl")
        for docket in JUDGMENT_DOCKETS:
            docs, _ = sp.load_docs(2022, docket, False)
            for rep in (1, 2):
                rec = cache.get(f"std|neutral|{docket}|{rep}|analysis")
                if not rec:
                    continue
                a = llm.parse_json(rec["text"]) or {}
                for field in ("focal_point", "petitioner_position", "respondent_position"):
                    q = a.get(field)
                    if not isinstance(q, str) or not q.strip():
                        continue
                    srcs = [d.metadata["source"] for d in sp.pipe_retrieve(docs, q, 5)]
                    rows.append(dict(model=model, docket=docket, rep=rep, query_field=field,
                                     judgment_chunk_retrieved=any(s in late for s in srcs)))
    # closer proxy for the justices' own queries ("the single legal concept or precedent you
    # would most want to examine"): the key precedents in the pipeline's saved case analyses
    runs = C.load_runs()
    for _, r in runs[(runs.folder == "term_2022_2023") & runs.docket.isin(JUDGMENT_DOCKETS)].iterrows():
        docs, _ = sp.load_docs(2022, r.docket, False)
        for q in (r.case_analysis.get("key_precedents") or [])[:9]:
            if isinstance(q, str) and q.strip():
                srcs = [d.metadata["source"] for d in sp.pipe_retrieve(docs, q, 5)]
                rows.append(dict(model=r.model, docket=r.docket, rep=r.replicate, query_field="pipeline_key_precedent",
                                 judgment_chunk_retrieved=any(s in late for s in srcs)))
    d = pd.DataFrame(rows)
    print(d.groupby("query_field").judgment_chunk_retrieved.agg(["size", "sum"]))
    d.to_csv(o / "research_exposure_proxy.csv", index=False)
    print(len(d), "retrievals;", int(d.judgment_chunk_retrieved.sum()), "retrieved a judgment chunk")
    print(d.groupby("docket").judgment_chunk_retrieved.sum())


if __name__ == "__main__":
    main()
