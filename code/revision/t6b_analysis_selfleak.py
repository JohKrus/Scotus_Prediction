"""Task 6 (part 2): outcome statements inside the pipeline's own case analyses.

The saved case analysis (step 1 of every run) is written before any justice
votes. Its inputs contain no opinion of the case (Task 1), so a statement of the
case's own outcome there comes from the model. This scans every saved analysis
for:
  - a U.S. Reports cite for the case itself (its caption's first party near
    "U.S." with a volume from the decision's own Term or later), or
  - outcome language about "the Court"/"Supreme Court" ("ultimately held",
    "the Court held", "reversed", "vacated", "affirmed") near the case's own name.
Hits are written for manual review, with counts by Term and model.

    python -m revision.t6b_analysis_selfleak
"""
from __future__ import annotations

import json
import re

import pandas as pd

from revision import common as C

# first U.S. Reports volume of each Term (approximate; volumes 598-600 = OT2022 etc.)
FIRST_VOLUME = {2022: 598, 2023: 601, 2024: 603, 2025: 606}
_OUTCOME = re.compile(r"(ultimately\s+(?:held|ruled|decided|reversed|affirmed|vacated)|"
                      r"(?:supreme\s+)?court\s+(?:held|ruled|reversed|affirmed|vacated)|"
                      r"\b(?:reversed|vacated|affirmed)\s+(?:and|the)\b)", re.I)


def party_token(case_name: str) -> str | None:
    first = re.split(r"\s+v\.?\s+", str(case_name), maxsplit=1)[0]
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z'\-]{3,}", first)
             if w.upper() not in {"UNITED", "STATES", "STATE", "INC", "CORP", "COMPANY", "DEPARTMENT", "CITY",
                                  "COUNTY", "NATIONAL", "AMERICAN", "ASSOCIATION", "FOUNDATION"}]
    return words[0] if words else None


def main():
    o = C.out("t6_fingerprints")
    runs = C.scored_runs()
    sc = C.scdb_cases()
    rows = []
    for _, r in runs.iterrows():
        text = json.dumps(r.case_analysis, ensure_ascii=False)
        tok = party_token(sc.caseName.get(r.docket, ""))
        if not tok:
            continue
        hits = []
        for m in re.finditer(re.escape(tok), text, re.I):
            win = text[max(0, m.start() - 40): m.end() + 260]
            v = re.search(r"\b(\d{3})\s*U\.\s*S\.", win)
            if v and int(v.group(1)) >= FIRST_VOLUME[r.term]:
                hits.append(("own_us_cite", win))
            elif _OUTCOME.search(win):
                hits.append(("outcome_language_near_name", win))
        for kind, win in hits[:3]:
            rows.append(dict(term=r.term, model=r.model, docket=r.docket, replicate=r.replicate, kind=kind,
                             correct=r.correct, snippet=win[:300]))
    d = pd.DataFrame(rows)
    d.to_csv(o / "analysis_selfleak_candidates.csv", index=False)
    if len(d):
        runs_hit = d.drop_duplicates(["model", "docket", "replicate"])
        tab = (runs_hit.groupby(["term", "model"]).size().rename("runs_with_candidate")
               .to_frame().join(runs.groupby(["term", "model"]).size().rename("runs")))
        tab["share"] = (tab.runs_with_candidate / tab.runs).round(3)
        tab.to_csv(o / "analysis_selfleak_counts.csv")
        print(tab.to_string())
        print(d[d.kind == "own_us_cite"][["term", "model", "docket", "snippet"]].head(20).to_string())


if __name__ == "__main__":
    main()
