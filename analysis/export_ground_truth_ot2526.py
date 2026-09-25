"""Export OT2025-26 justice-level ground truth in the format of code/data/ground_truth/.

OT2025-26 is not in the SCDB yet; the source is the corrected results sheet
(scotus_ot2025_cases.xlsx), parsed by parse_justice_votes_ot2526. Columns follow
justice_votes_2022_term.csv, whose ActualVote column scotus_v2/evaluate.py reads
directly: Term, NaturalCourt, Docket, CaseName, DecisionDate, Justice,
VoteDescription, ActualVote, WinningParty. A recused justice gets no row.

Run:  python export_ground_truth_ot2526.py [out.csv]
"""
from __future__ import annotations

import csv
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from paths import DATA_DIR  # noqa: E402

from parse_justice_votes_ot2526 import load_justice_truth, load_writings_truth  # noqa: E402

SCDB_CODE = {"Roberts": "JGRoberts", "Thomas": "CThomas", "Alito": "SAAlito",
             "Sotomayor": "SSotomayor", "Kagan": "EKagan", "Gorsuch": "NMGorsuch",
             "Kavanaugh": "BMKavanaugh", "Barrett": "ACBarrett", "Jackson": "KBJackson"}
ROLE_TEXT = {"author": "Majority or plurality", "join": "Majority or plurality",
             "concur": "Regular concurrence", "concur_judgment": "Special concurrence",
             "dissent": "Dissent"}
WINNER_TEXT = {"Petitioner": "Petitioner/Appellant won (petitioning party received a favorable disposition)",
               "Respondent": "Respondent/Appellee won (petitioning party did not receive a favorable disposition)"}


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "justice_votes_2025_2026.csv")
    x = pd.read_excel(os.path.join(DATA_DIR, "scotus_ot2025_cases.xlsx"))
    meta = {str(r["Docket number"]).strip(): r for _, r in x.iterrows()}
    votes, _ = load_justice_truth()
    writings = load_writings_truth()
    rows = []
    for d in sorted(votes):
        r = meta[d]
        for sur, party in votes[d].items():
            rows.append({
                "Term": 2025, "NaturalCourt": 1710, "Docket": d, "CaseName": r["Case name"],
                "DecisionDate": pd.to_datetime(r["Date decided"]).strftime("%-m/%-d/%Y")
                if os.name != "nt" else pd.to_datetime(r["Date decided"]).strftime("%#m/%#d/%Y"),
                "Justice": SCDB_CODE[sur],
                "VoteDescription": ROLE_TEXT[writings[d]["roles"][sur]],
                "ActualVote": party,
                "WinningParty": WINNER_TEXT[r["Winner (petitioner/respondent)"]],
            })
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows for {len(votes)} cases to {out}")


if __name__ == "__main__":
    main()
