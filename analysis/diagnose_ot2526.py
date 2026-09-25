"""Diagnose why the OT2025-26 forecasts underperformed the majority-class baseline.

Decomposes the error into (a) the persona-level signal, (b) the aggregation rule
that turns nine persona votes into a case call, and (c) the deliberation step,
using only the saved prediction artifacts plus the parsed OT2025-26 justice-level
truth. No API calls.

Run from repo root:  python v2/code/diagnose_ot2526.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
from paths import DATA_DIR, PRED_DIR  # noqa: E402
sys.path.insert(0, HERE)

from parse_justice_votes_ot2526 import PERSONA_TO_SURNAME, load_justice_truth  # noqa: E402
from score_predictions import wilson  # noqa: E402

TERMS = {"2022_2023": 2022, "2023_2024": 2023, "2024_2025": 2024, "2025_2026": 2025}


def load_preds(term_key: str) -> list[dict]:
    out = []
    for f in glob.glob(os.path.join(PRED_DIR, f"term_{term_key}", "*.json")):
        try:
            out.extend(json.load(open(f, encoding="utf-8")))
        except Exception:
            continue
    return out


def ot2526_case_truth() -> dict[str, str]:
    x = pd.read_excel(os.path.join(DATA_DIR, "scotus_ot2025_cases.xlsx"))
    t = {}
    for _, r in x.iterrows():
        w = r["Winner (petitioner/respondent)"]
        if isinstance(w, str) and w.strip():
            t[str(r["Docket number"]).strip()] = w.strip()
    return t


def hdr(s: str) -> None:
    print("\n" + "=" * 76 + f"\n{s}\n" + "=" * 76)


def main() -> None:
    truth = ot2526_case_truth()
    jtruth, _ = load_justice_truth()
    preds = [p for p in load_preds("2025_2026") if p["docket"] in truth]

    # ---------------------------------------------------------------- 1
    hdr("1. IS THE MODEL TOO UNANIMOUS?  predicted vs actual vote splits")
    x = pd.read_excel(os.path.join(DATA_DIR, "scotus_ot2025_cases.xlsx"))
    actual_split = {str(r["Docket number"]).strip(): r["Vote split"]
                    for _, r in x.iterrows() if isinstance(r["Vote split"], str)}
    pc, ac = defaultdict(int), defaultdict(int)
    for p in preds:
        pc[p.get("vote_split")] += 1
    for d in {p["docket"] for p in preds}:
        if d in actual_split:
            ac[actual_split[d]] += 1
    npred, nact = sum(pc.values()), sum(ac.values())
    print(f"  {'split':>7} {'predicted':>12} {'actual':>12}")
    for s in ["9-0", "8-1", "7-2", "6-3", "5-4", "8-0"]:
        if pc.get(s) or ac.get(s):
            print(f"  {s:>7} {100*pc.get(s,0)/npred:11.1f}% {100*ac.get(s,0)/nact:11.1f}%")
    lop = sum(pc.get(s, 0) for s in ("9-0", "8-1")) / npred
    lopa = sum(ac.get(s, 0) for s in ("9-0", "8-1", "8-0")) / nact
    print(f"\n  lopsided (9-0/8-1): predicted {100*lop:.1f}%  vs actual {100*lopa:.1f}%")

    # ---------------------------------------------------------------- 2
    hdr("2. PERSONA-LEVEL SIGNAL: are the individual justice votes any good?")
    jt_ok = [p for p in preds if p["docket"] in jtruth]
    k = n = 0
    per_j = defaultdict(lambda: [0, 0])
    per_j_pred_pet = defaultdict(lambda: [0, 0])
    for p in jt_ok:
        for persona, jv in p.get("justice_votes", {}).items():
            sur = PERSONA_TO_SURNAME.get(persona)
            tv = jtruth[p["docket"]].get(sur)
            if tv is None:
                continue
            ok = int(jv.get("vote") == tv)
            k += ok; n += 1
            per_j[sur][0] += ok; per_j[sur][1] += 1
            per_j_pred_pet[sur][0] += int(jv.get("vote") == "Petitioner")
            per_j_pred_pet[sur][1] += 1
    p_, lo, hi = wilson(k, n)
    print(f"  Justice-vote accuracy: {p_:.1f}%  [{lo:.1f}, {hi:.1f}]  (n={n} votes, {len(jtruth)} cases)")
    print(f"\n  {'justice':<11} {'acc':>7}   {'pred pet':>9}  {'true pet':>9}   gap")
    for sur in ["Roberts", "Thomas", "Alito", "Sotomayor", "Kagan",
                "Gorsuch", "Kavanaugh", "Barrett", "Jackson"]:
        kk, nn = per_j[sur]
        pp = 100 * per_j_pred_pet[sur][0] / per_j_pred_pet[sur][1]
        seated = [v[sur] for v in jtruth.values() if sur in v]
        tp = 100 * sum(tv == "Petitioner" for tv in seated) / len(seated)
        print(f"  {sur:<11} {100*kk/nn:6.1f}%   {pp:8.1f}% {tp:8.1f}%   {pp-tp:+6.1f}")

    # ---------------------------------------------------------------- 3
    hdr("3. AGREEMENT STRUCTURE: do the personas differentiate from each other?")
    # how often do all 9 personas vote identically, vs the real Court
    unan_pred = sum(1 for p in jt_ok
                    if len({v.get("vote") for v in p.get("justice_votes", {}).values()}) == 1)
    print(f"  Runs where all 9 personas agree: {100*unan_pred/len(jt_ok):.1f}%")
    real_unan = sum(1 for d in jtruth if len(set(jtruth[d].values())) == 1)
    print(f"  Cases where the real Court was unanimous: {100*real_unan/len(jtruth):.1f}%")
    # pairwise: does persona disagreement track real disagreement?
    rows = []
    for p in jt_ok:
        jv = p.get("justice_votes", {})
        pred_min = min(sum(v.get("vote") == "Petitioner" for v in jv.values()),
                       sum(v.get("vote") == "Respondent" for v in jv.values()))
        t = jtruth[p["docket"]]
        true_min = min(sum(v == "Petitioner" for v in t.values()),
                       sum(v == "Respondent" for v in t.values()))
        rows.append((pred_min, true_min))
    df = pd.DataFrame(rows, columns=["pred_dissent", "true_dissent"])
    print(f"  Mean predicted dissenters: {df.pred_dissent.mean():.2f}   "
          f"actual: {df.true_dissent.mean():.2f}")
    print(f"  Correlation(predicted dissent count, actual): {df.corr().iloc[0,1]:+.3f}")

    # ---------------------------------------------------------------- 4
    hdr("4. DELIBERATION: does it move anything, and in which direction?")
    for term_key, term in TERMS.items():
        tp = load_preds(term_key)
        if term == 2025:
            tt = truth
        else:
            sc = json.load(open(os.path.join(HERE, "cache", "scdb_truth.json")))
            tt = sc["case_winner"]
        tp = [p for p in tp if p["docket"] in tt]
        if not tp:
            continue
        ki = sum(p.get("initial_winner") == tt[p["docket"]] for p in tp)
        kf = sum(p.get("predicted_winner") == tt[p["docket"]] for p in tp)
        moved = sum(p.get("initial_winner") != p.get("predicted_winner") for p in tp)
        # of those that moved, how many moved to the right answer
        good = sum(1 for p in tp if p.get("initial_winner") != p.get("predicted_winner")
                   and p.get("predicted_winner") == tt[p["docket"]])
        print(f"  OT{term}: initial {100*ki/len(tp):5.1f}% -> final {100*kf/len(tp):5.1f}%  "
              f"| moved {moved:3d}/{len(tp)} runs, {good} of those toward truth")

    # ---------------------------------------------------------------- 5
    hdr("5. WHERE THE LOSSES ARE: accuracy by true margin (OT2025-26)")
    by_margin = defaultdict(lambda: [0, 0])
    for p in preds:
        s = actual_split.get(p["docket"])
        if not s:
            continue
        by_margin[s][0] += int(p["predicted_winner"] == truth[p["docket"]])
        by_margin[s][1] += 1
    for s in sorted(by_margin, key=lambda z: -by_margin[z][1]):
        kk, nn = by_margin[s]
        print(f"  {s:>5}: {100*kk/nn:5.1f}%  (n={nn} runs)")

    hdr("6. DIRECTIONAL BIAS BY TERM (predicted petitioner rate vs true rate)")
    sc = json.load(open(os.path.join(HERE, "cache", "scdb_truth.json")))
    for term_key, term in TERMS.items():
        tt = truth if term == 2025 else sc["case_winner"]
        tp = [p for p in load_preds(term_key) if p["docket"] in tt]
        if not tp:
            continue
        pred_pet = 100 * sum(p["predicted_winner"] == "Petitioner" for p in tp) / len(tp)
        dk = {p["docket"] for p in tp}
        true_pet = 100 * sum(tt[d] == "Petitioner" for d in dk) / len(dk)
        print(f"  OT{term}: predicted petitioner {pred_pet:5.1f}%  |  actual {true_pet:5.1f}%  "
              f"| bias {pred_pet-true_pet:+5.1f}")


if __name__ == "__main__":
    main()
