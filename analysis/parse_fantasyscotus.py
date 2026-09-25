"""Parse a FantasySCOTUS OT2025-26 case-list export into ot2526_outcomes.csv.

Block format (variable length) per case in the export:
  case number, case name, date argued, [date decided],
  crowd_direction, crowd_split, [outcome_direction, outcome_split | "Predict"], "View"
A case is DECIDED-with-outcome iff it has TWO Affirm/Reverse direction tokens;
the second is the actual outcome. One direction followed by "Predict" => not yet
scored. Mapping: Reverse -> Petitioner prevails, Affirm -> Respondent prevails
(petitioner seeks reversal; matches SCDB partyWinning).

Outcomes come from the corrected results sheet (scotus_ot2025_cases.xlsx); the
export supplies only the crowd's pick. Writes one row per decided docket in our
OT2025-26 predictions, with fantasyscotus_correct blank where the case is not on
FantasySCOTUS (unargued per curiam decisions).

Usage:  python v2/code/parse_fantasyscotus.py [path/to/export.txt]   (default ~/Downloads/scotus_ot_25_26.txt)
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
from paths import DATA_DIR, PRED_DIR  # noqa: E402
PRED = os.path.join(PRED_DIR, "term_2025_2026", "*.json")
SHEET = os.path.join(DATA_DIR, "scotus_ot2025_cases.xlsx")
OUT = os.path.join(HERE, "ot2526_outcomes.csv")

# Dockets have a 2+ digit term prefix (23/24/25...); vote splits like "6-3" have a
# single-digit prefix and must NOT be treated as dockets.
DOCKET_RE = re.compile(r"^\d{2,}-\d+$|^\d{2,}a\d+$", re.I)
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
SPLIT_RE = re.compile(r"^\d-\d$")
DIR_RE = re.compile(r"^(Affirm|Reverse)$", re.I)
DOCKET_TOKEN_RE = re.compile(r"\b\d{2,}-\d+\b|\b\d{2,}A\d+\b", re.I)


def norm(dk: str) -> str:
    return dk.strip().upper()


def our_dockets() -> set[str]:
    s = set()
    for f in glob.glob(PRED):
        for d in json.load(open(f, encoding="utf-8")):
            s.add(norm(d["docket"]))
    return s


def parse(path: str):
    lines = [ln.strip() for ln in open(path, encoding="utf-8")]
    # find case-number line indices (skip the header block at top)
    idxs = [i for i, ln in enumerate(lines) if DOCKET_RE.match(ln)]
    cases = []
    for j, start in enumerate(idxs):
        end = idxs[j + 1] if j + 1 < len(idxs) else len(lines)
        block = lines[start:end]
        docket = block[0]
        dirs = [b for b in block if DIR_RE.match(b)]
        decided = len(dirs) >= 2  # two directions => outcome present
        crowd = dirs[0].capitalize() if dirs else None
        outcome = dirs[1].capitalize() if decided else None
        cases.append({"docket": docket, "crowd": crowd, "outcome": outcome,
                      "decided": decided})
    return cases


def to_winner(direction: str) -> str:
    return "Petitioner" if direction.lower() == "reverse" else "Respondent"


def sheet_outcomes() -> tuple[dict[str, str], dict[str, str]]:
    """(docket -> winner, companion docket -> lead docket) from the corrected results sheet.

    The sheet is the authoritative outcome source; FantasySCOTUS supplies only the
    crowd's pick. FantasySCOTUS sometimes lists a consolidated case under its
    companion docket (Callais as 24-110), hence the alias map.
    """
    x = pd.read_excel(SHEET)
    winners, alias = {}, {}
    for _, r in x.iterrows():
        dk = norm(str(r["Docket number"]))
        if isinstance(r["Winner (petitioner/respondent)"], str):
            winners[dk] = r["Winner (petitioner/respondent)"].strip()
        for comp in DOCKET_TOKEN_RE.findall(str(r["Consolidated dockets"])):
            alias[norm(comp)] = dk
    return winners, alias


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/Downloads/scotus_ot_25_26.txt")
    ours = our_dockets()
    winners, alias = sheet_outcomes()
    fs = {}
    for c in parse(path):
        dk = norm(c["docket"])
        fs.setdefault(alias.get(dk, dk), c)   # a lead-docket entry wins over a companion

    scored = sorted(dk for dk in ours if dk in winners)
    print(f"FantasySCOTUS cases parsed: {len(fs)} | our decided dockets: {len(scored)}")
    print("\nDocket     crowd    FS outcome  sheet winner  crowd_correct  basis")
    rows = []
    for dk in scored:
        c = fs.get(dk)
        if c is None or c["crowd"] is None:
            rows.append((dk, winners[dk], ""))
            print(f"  {dk:9s} {'-':7s}  {'-':10s}  {winners[dk]:12s}  {'':13s}  not on FantasySCOTUS")
            continue
        if c["decided"]:
            # Score within FantasySCOTUS's own party alignment: for consolidated
            # cases it may code direction from the companion case's posture
            # (Learning Resources: "Affirm" = tariffs struck down).
            ok = int(c["crowd"] == c["outcome"])
            basis = "FS outcome"
            if to_winner(c["outcome"]) != winners[dk]:
                basis += "  (FS alignment differs from sheet)"
        else:
            # Undecided when the export was taken: the crowd's pick at that date.
            ok = int(to_winner(c["crowd"]) == winners[dk])
            basis = "sheet (pick as of export date)"
        rows.append((dk, winners[dk], ok))
        print(f"  {dk:9s} {c['crowd']:7s}  {str(c['outcome'] or '-'):10s}  {winners[dk]:12s}  {ok:<13d}  {basis}")

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("docket,winner,fantasyscotus_correct\n")
        for dk, win, fsc in rows:
            fh.write(f"{dk},{win},{fsc}\n")
    n = sum(1 for r in rows if r[2] != "")
    print(f"\nWrote {len(rows)} rows to {OUT} ({n} with a crowd pick, "
          f"{sum(r[2] for r in rows if r[2] != '')} correct)")


if __name__ == "__main__":
    main()
