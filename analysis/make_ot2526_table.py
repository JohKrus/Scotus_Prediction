"""Emit the OT2025-26 exact-predictions appendix table (LaTeX longtable body).

Writes paper/ot2526_predictions_table.tex, \\input by the appendix. One row per
docket: each model's consensus predicted winner and modal vote split across the
two replicates (a dagger marks dockets where a model's two replicates disagree
on the winner). Regenerable; no web/API needed.

Run from repo root:  python v2/code/make_ot2526_table.py
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict, Counter

from paths import PAPER_DIR, PRED_DIR  # noqa: E402
PRED = os.path.join(PRED_DIR, "term_2025_2026", "*.json")
OUT = os.path.join(PAPER_DIR, "ot2526_predictions_table.tex")
MODELS = ["Claude-4.6", "GPT-5.2"]


def main():
    data = defaultdict(lambda: defaultdict(list))  # docket -> model -> [(winner, split)]
    for f in glob.glob(PRED):
        for d in json.load(open(f, encoding="utf-8")):
            data[d["docket"]][d["llm_model"]].append((d["predicted_winner"], d["vote_split"]))

    lines = []
    for docket in sorted(data):
        cells = []
        for m in MODELS:
            reps = data[docket].get(m, [])
            if not reps:
                cells.append("---"); continue
            winners = [w for w, _ in reps]
            modal_w = Counter(winners).most_common(1)[0][0]
            split = Counter(s for _, s in reps).most_common(1)[0][0]
            disagree = "$^{\\dagger}$" if len(set(winners)) > 1 else ""
            w_abbr = "Pet." if modal_w == "Petitioner" else "Resp."
            cells.append(f"{w_abbr} ({split}){disagree}")
        # escape underscores in docket if any (dockets like 24A884 are fine)
        lines.append(f"{docket} & {cells[0]} & {cells[1]} \\\\")

    body = "\n".join(lines)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(body + "\n")
    print(f"Wrote {len(lines)} rows to {OUT}")


if __name__ == "__main__":
    main()
