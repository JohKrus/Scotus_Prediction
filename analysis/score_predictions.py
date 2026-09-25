"""Score v2 deliberation predictions against the SCDB ground truth.

Outputs case-level and justice-level accuracy by model and term, the
deliberation effect (initial vs. final), a majority-class baseline, and a
calibration table. OT 2025-26 has no SCDB ground truth and is skipped for
accuracy (reported as pending elsewhere).

Run from the repo root:  python v2/code/score_predictions.py
"""
from __future__ import annotations

import glob
import json
import math
import os
import re
from collections import defaultdict

import pandas as pd

from paths import DATA_DIR, PRED_DIR  # noqa: E402
CASE_CSV = os.path.join(DATA_DIR, "SCDB_2025_01_caseCentered_Citation.csv")
JUSTICE_CSV = os.path.join(DATA_DIR, "SCDB_2025_01_justiceCentered_Citation.csv")
PRED_GLOB = os.path.join(PRED_DIR, "term_*", "*_predictions.json")

# prediction folder term -> SCDB term (OT year)
TERM_MAP = {"2022_2023": 2022, "2023_2024": 2023, "2024_2025": 2024, "2025_2026": 2025}

# prediction full name -> SCDB justiceName code
JUSTICE_MAP = {
    "John Roberts": "JGRoberts", "Clarence Thomas": "CThomas", "Samuel Alito": "SAAlito",
    "Sonia Sotomayor": "SSotomayor", "Elena Kagan": "EKagan", "Neil Gorsuch": "NMGorsuch",
    "Brett Kavanaugh": "BMKavanaugh", "Amy Coney Barrett": "ACBarrett",
    "Ketanji Brown Jackson": "KBJackson",
}


def wilson(k: int, n: int) -> tuple[float, float, float]:
    """Return (point, lo, hi) Wilson 95% interval as percentages."""
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    z = 1.959963985
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (100 * p, 100 * (center - half), 100 * (center + half))


def modal_winner(preds: list[str]) -> str:
    """Modal predicted winner across runs; a tie goes to Petitioner (the majority class).

    Deterministic by design: max(set(preds), key=preds.count) breaks ties by set
    iteration order, which varies with Python's per-process string hashing.
    """
    pet = sum(p == "Petitioner" for p in preds)
    res = sum(p == "Respondent" for p in preds)
    return "Petitioner" if pet >= res else "Respondent"


def load_scdb():
    cc = pd.read_csv(CASE_CSV, encoding="latin-1", low_memory=False)
    jc = pd.read_csv(JUSTICE_CSV, encoding="latin-1", low_memory=False)
    # case winner: 1 -> petitioner, 0 -> respondent; drop unclear/nan
    # keyed by docket; also record the SCDB term (decided term) for grouping
    case_winner = {}
    case_term = {}
    for _, r in cc.iterrows():
        pw = r["partyWinning"]
        if pw in (0.0, 1.0):
            d = str(r["docket"])
            case_winner[d] = "Petitioner" if pw == 1.0 else "Respondent"
            case_term[d] = int(r["term"]) if pd.notna(r["term"]) else None
    load_scdb.case_term = case_term
    # per-term petitioner base rate (for "always petitioner / always reverse" baseline)
    base = {}
    for term in (2022, 2023, 2024):
        sub = cc[(cc.term == term) & (cc.partyWinning.isin([0.0, 1.0]))]
        base[term] = (sub.partyWinning == 1.0).mean() if len(sub) else float("nan")
    # justice votes: in-majority XOR winner -> petitioner/respondent vote
    # voted_petitioner == (in_majority == petitioner_won)
    justice_vote = {}  # (docket, justiceCode) -> "Petitioner"/"Respondent"
    for _, r in jc.iterrows():
        d = str(r["docket"]); maj = r["majority"]
        win = case_winner.get(d)
        if win is None or maj not in (1.0, 2.0):
            continue
        in_majority = maj == 2.0
        petitioner_won = win == "Petitioner"
        justice_vote[(d, str(r["justiceName"]))] = (
            "Petitioner" if in_majority == petitioner_won else "Respondent"
        )
    return case_winner, justice_vote, base


def main():
    case_winner, justice_vote, base = load_scdb()
    case_term = load_scdb.case_term

    # accumulators keyed by (term, model)
    case_final = defaultdict(lambda: [0, 0])    # [correct, n]   (replicate-level)
    case_initial = defaultdict(lambda: [0, 0])
    just_final = defaultdict(lambda: [0, 0])
    just_by_justice = defaultdict(lambda: [0, 0])  # (term, model, justice)
    winner_changed = defaultdict(lambda: [0, 0])
    flips_total = defaultdict(lambda: [0.0, 0])
    calib = defaultdict(lambda: [0, 0])  # confidence-bin -> [correct, n] (all models, scoreable terms)
    consensus = defaultdict(lambda: [0, 0])  # (term, model) docket-level modal prediction
    unmatched = defaultdict(int)

    consensus_acc = defaultdict(list)  # (term,model) -> per docket (preds list, truth)
    docket_preds = defaultdict(list)   # (term, model, docket) -> list of predicted_winner

    for f in glob.glob(PRED_GLOB):
        folder = os.path.basename(os.path.dirname(f))
        term_key = folder.replace("term_", "")
        try:
            arr = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        for d in arr:
            docket = d.get("docket"); model = d.get("llm_model")
            truth = case_winner.get(docket)
            scdb_term = case_term.get(docket)        # group by actual SCDB term
            if truth is None or scdb_term is None:
                unmatched[(term_key, model)] += 1
                continue
            key = (scdb_term, model)
            # case-level final
            pred = d.get("predicted_winner")
            case_final[key][1] += 1
            case_final[key][0] += int(pred == truth)
            docket_preds[(scdb_term, model, docket)].append(pred)
            # case-level initial (deliberation effect)
            init = d.get("initial_winner")
            case_initial[key][1] += 1
            case_initial[key][0] += int(init == truth)
            winner_changed[key][1] += 1
            winner_changed[key][0] += int(init != pred)
            flips_total[key][0] += d.get("total_vote_changes", 0)
            flips_total[key][1] += 1
            # justice-level
            for jname, jv in d.get("justice_votes", {}).items():
                code = JUSTICE_MAP.get(jname)
                tv = justice_vote.get((docket, code))
                if tv is None:
                    continue
                pv = jv.get("vote")
                correct = int(pv == tv)
                just_final[key][0] += correct; just_final[key][1] += 1
                just_by_justice[(scdb_term, model, jname)][0] += correct
                just_by_justice[(scdb_term, model, jname)][1] += 1
                conf = jv.get("confidence")
                if isinstance(conf, (int, float)):
                    b = min(int(conf * 10), 9)
                    calib[b][0] += correct; calib[b][1] += 1

    # consensus (docket-level modal prediction; tie -> Petitioner)
    for (term, model, docket), preds in docket_preds.items():
        truth = case_winner.get(docket)
        modal = modal_winner(preds)
        consensus[(term, model)][1] += 1
        consensus[(term, model)][0] += int(modal == truth)

    def line(label, k, n):
        p, lo, hi = wilson(k, n)
        return f"  {label:28s} {p:5.1f}%  [{lo:4.1f}, {hi:4.1f}]   (n={n})"

    print("=" * 78)
    print("CASE-LEVEL ACCURACY (final, replicate-level) vs SCDB")
    print("=" * 78)
    for (term, model) in sorted(case_final):
        k, n = case_final[(term, model)]
        print(line(f"OT{term} {model}", k, n))
    print("\nBaseline 'always Petitioner' (= majority class / reversal):")
    for term in (2022, 2023, 2024):
        print(f"  OT{term}: {100*base[term]:.1f}%")

    print("\n" + "=" * 78)
    print("CASE-LEVEL ACCURACY (docket-level consensus of 2 replicates)")
    print("=" * 78)
    for (term, model) in sorted(consensus):
        k, n = consensus[(term, model)]
        print(line(f"OT{term} {model}", k, n))

    print("\n" + "=" * 78)
    print("DELIBERATION EFFECT")
    print("=" * 78)
    for (term, model) in sorted(case_final):
        ki, ni = case_initial[(term, model)]
        kf, nf = case_final[(term, model)]
        wc, wn = winner_changed[(term, model)]
        fl, fn = flips_total[(term, model)]
        print(f"  OT{term} {model}: initial {100*ki/ni:.1f}% -> final {100*kf/nf:.1f}%  "
              f"| winner changed in {100*wc/wn:.1f}% of runs | mean flips/case {fl/fn:.2f}")

    print("\n" + "=" * 78)
    print("JUSTICE-LEVEL ACCURACY (replicate-level)")
    print("=" * 78)
    for (term, model) in sorted(just_final):
        k, n = just_final[(term, model)]
        print(line(f"OT{term} {model}", k, n))

    print("\n" + "=" * 78)
    print("JUSTICE-LEVEL ACCURACY BY JUSTICE (pooled over scoreable terms)")
    print("=" * 78)
    pooled = defaultdict(lambda: [0, 0])
    for (term, model, jname), (k, n) in just_by_justice.items():
        pooled[(model, jname)][0] += k; pooled[(model, jname)][1] += n
    for model in sorted(set(m for m, _ in pooled)):
        print(f"  -- {model} --")
        for jname in JUSTICE_MAP:
            k, n = pooled[(model, jname)]
            if n:
                print(f"     {jname:24s} {100*k/n:5.1f}%  (n={n})")

    print("\n" + "=" * 78)
    print("CALIBRATION (justice-vote confidence vs empirical accuracy, all models)")
    print("=" * 78)
    for b in range(10):
        k, n = calib[b]
        if n:
            print(f"  conf [{b/10:.1f},{(b+1)/10:.1f}): empirical {100*k/n:5.1f}%  (n={n})")

    print("\nUnmatched-to-SCDB predictions (docket not in SCDB / no clear winner):")
    for (term_key, model), c in sorted(unmatched.items()):
        print(f"  term_{term_key} {model}: {c}")


if __name__ == "__main__":
    main()
