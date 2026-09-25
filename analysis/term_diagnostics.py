"""Diagnostics for the OT2024-as-clean-test interpretation, SCDB-only (no API).

(A) Term-composition confound: is OT2024 simply a harder/more-divided Term than
    OT2022-2023? Compares petitioner-win rate, vote-split distribution, and
    issue-area mix across Terms.

(B) Within-OT2024 decision-date split: if a training cutoff bisects OT2024
    (decided Nov 2024 - Jun 2025), case-level accuracy should fall for
    later-decided cases. Bins OT2024 accuracy by decision date and reports a
    trend test. Low power (~64 cases) -- suggestive, not dispositive.

Run from repo root:  python v2/code/term_diagnostics.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_predictions import load_scdb, wilson, PRED_GLOB, CASE_CSV  # noqa: E402

ISSUE_AREA = {
    1: "Criminal Procedure", 2: "Civil Rights", 3: "First Amendment",
    4: "Due Process", 5: "Privacy", 6: "Attorneys", 7: "Unions",
    8: "Economic Activity", 9: "Judicial Power", 10: "Federalism",
    11: "Interstate Relations", 12: "Federal Taxation", 13: "Misc", 14: "Private Action",
}


def load_case_meta():
    cc = pd.read_csv(CASE_CSV, encoding="latin-1", low_memory=False)
    cc["d"] = pd.to_datetime(cc["dateDecision"], errors="coerce")
    return cc


def part_a(cc):
    print("=" * 80)
    print("(A) TERM-COMPOSITION CONFOUND")
    print("=" * 80)
    print(f"  {'Term':6s} {'n':>4s} {'Pet-win%':>9s} {'9-0%':>7s} {'minVotes>=4 (close)%':>20s} {'med opinion gap':>16s}")
    for t in (2022, 2023, 2024):
        sub = cc[cc.term == t]
        n = len(sub)
        petwin = 100 * (sub.partyWinning == 1).mean()
        unanimous = 100 * (sub.minVotes == 0).mean()
        close = 100 * (sub.minVotes >= 4).mean()
        print(f"  OT{t} {n:>4d} {petwin:8.1f}% {unanimous:6.1f}% {close:19.1f}%")
    print("\n  Issue-area mix (% of Term's cases):")
    areas = sorted(set(cc[cc.term.isin([2022, 2023, 2024])].issueArea.dropna().astype(int)))
    header = "    " + f"{'Issue area':24s}" + "".join(f"OT{t:>6d}" for t in (2022, 2023, 2024))
    print(header)
    for a in areas:
        row = f"    {ISSUE_AREA.get(a, str(a)):24s}"
        for t in (2022, 2023, 2024):
            sub = cc[cc.term == t]
            pct = 100 * (sub.issueArea == a).mean()
            row += f"{pct:7.1f}"
        print(row)
    print("\n  Reading: if OT2024's drop were just difficulty, OT2024 should be far more")
    print("  divided than OT2022-23. Compare the close-case shares and issue mix above.")


def part_b(cc):
    case_winner, _, _ = load_scdb()
    # decision date per docket (OT2024 only)
    ot24 = cc[cc.term == 2024][["docket", "d"]].dropna()
    date_of = {str(r.docket): r.d for r in ot24.itertuples()}

    # collect replicate-level correctness for OT2024, pooled over Claude+GPT
    recs = []  # (date, correct)
    for f in glob.glob(PRED_GLOB):
        try:
            arr = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        for d in arr:
            docket = d.get("docket")
            if docket not in date_of:
                continue
            truth = case_winner.get(docket)
            if truth is None:
                continue
            recs.append((date_of[docket], int(d.get("predicted_winner") == truth)))

    print("\n" + "=" * 80)
    print("(B) WITHIN-OT2024 ACCURACY BY DECISION DATE  (pooled Claude+GPT, replicate-level)")
    print("=" * 80)
    if not recs:
        print("  No OT2024 records matched.")
        return
    dates = np.array([pd.Timestamp(r[0]).value for r in recs], dtype=float)
    correct = np.array([r[1] for r in recs])
    order = np.argsort(dates)

    # coarse buckets by decision period
    buckets = [
        ("Nov 2024 - Feb 2025", "2024-11-01", "2025-03-01"),
        ("Mar 2025 - Apr 2025", "2025-03-01", "2025-05-01"),
        ("May 2025 - Jun 2025", "2025-05-01", "2025-07-01"),
    ]
    for label, lo, hi in buckets:
        lo_v, hi_v = pd.Timestamp(lo).value, pd.Timestamp(hi).value
        mask = (dates >= lo_v) & (dates < hi_v)
        k, n = int(correct[mask].sum()), int(mask.sum())
        if n:
            p, plo, phi = wilson(k, n)
            print(f"  {label:22s} acc {p:5.1f}%  [{plo:4.1f}, {phi:4.1f}]  (n={n})")
        else:
            print(f"  {label:22s} (no cases)")

    # trend test: point-biserial correlation of decision date vs correctness
    if len(set(correct)) > 1:
        r = np.corrcoef(dates, correct)[0, 1]
        print(f"\n  corr(decision date, correct) = {r:+.3f}  "
              "(negative => later-decided cases predicted worse, consistent with a"
              "\n   cutoff bisecting the Term; magnitude small and n modest -- suggestive only)")


def part_c(cc):
    """OT2024 accuracy split by issue area, to test whether the drop is driven by
    the OT2024 spike in Judicial Power cases (putatively *easier* per Stiglitz)."""
    case_winner, _, _ = load_scdb()
    issue_of = {str(r.docket): r.issueArea for r in cc[cc.term == 2024].itertuples()}
    by_judpow = defaultdict(lambda: [0, 0])  # "Judicial Power" / "Other" -> [correct, n]
    for f in glob.glob(PRED_GLOB):
        try:
            arr = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        for d in arr:
            docket = d.get("docket")
            ia = issue_of.get(docket)
            truth = case_winner.get(docket)
            if ia is None or truth is None or pd.isna(ia):
                continue
            grp = "Judicial Power" if int(ia) == 9 else "Other"
            by_judpow[grp][0] += int(d.get("predicted_winner") == truth)
            by_judpow[grp][1] += 1
    print("\n" + "=" * 80)
    print("(C) OT2024 ACCURACY BY ISSUE GROUP  (pooled Claude+GPT, replicate-level)")
    print("=" * 80)
    for grp in ("Judicial Power", "Other"):
        k, n = by_judpow[grp]
        if n:
            p, lo, hi = wilson(k, n)
            print(f"  {grp:16s} acc {p:5.1f}%  [{lo:4.1f}, {hi:4.1f}]  (n={n})")
    print("  (If the OT2024 drop were a composition artifact of more Judicial-Power")
    print("   cases, those cases would have to be where accuracy is lowest.)")


def main():
    cc = load_case_meta()
    part_a(cc)
    part_b(cc)
    part_c(cc)


if __name__ == "__main__":
    main()
