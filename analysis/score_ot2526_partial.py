"""Partial scoring of the pre-registered OT2025-26 forecasts.

OT2025-26 is not in the SCDB yet, and the decided-case outcomes could not be
scraped automatically (FantasySCOTUS and the court trackers are JS-rendered).
This script therefore scores our forecasts against a small, hand-maintained
outcomes file that the user populates as cases are decided (e.g., from the
FantasySCOTUS case list or supremecourt.gov):

    v2/code/ot2526_outcomes.csv
        docket,winner[,fantasyscotus_correct]
        24-1063,Respondent,1
        ...
    winner in {Petitioner, Respondent}; fantasyscotus_correct optional (1/0)
    for whether the FantasySCOTUS crowd called that case correctly.

It prints case-level accuracy (replicate-level and docket consensus) on the
decided subset, with a Wilson interval, and the FantasySCOTUS crowd accuracy on
the same subset if provided. Once the SCDB covers OT2025, switch to
score_predictions.py for the full, authoritative scoring.

Run from repo root:  python v2/code/score_ot2526_partial.py
"""
from __future__ import annotations

import csv
import glob
import json
import os
from collections import defaultdict

from score_predictions import modal_winner, wilson  # noqa: E402  (same dir on sys.path when run from repo root)

HERE = os.path.dirname(os.path.abspath(__file__))
from paths import PRED_DIR  # noqa: E402
OUTCOMES = os.path.join(HERE, "ot2526_outcomes.csv")
PRED = os.path.join(PRED_DIR, "term_2025_2026", "*.json")


def main():
    if not os.path.exists(OUTCOMES):
        print(f"No outcomes file at {OUTCOMES}. Populate it as cases are decided.")
        return
    truth, fs_correct = {}, {}
    with open(OUTCOMES, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            d = row["docket"].strip()
            truth[d] = row["winner"].strip()
            if row.get("fantasyscotus_correct", "").strip() != "":
                fs_correct[d] = int(row["fantasyscotus_correct"])
    if not truth:
        print("Outcomes file is empty.")
        return

    rep = [0, 0]                              # replicate-level [correct, n]
    by_model = defaultdict(lambda: [0, 0])
    docket_preds = defaultdict(lambda: defaultdict(list))  # docket -> model -> [winner]
    for f in glob.glob(PRED):
        for d in json.load(open(f, encoding="utf-8")):
            dk = d.get("docket")
            if dk not in truth:
                continue
            ok = int(d.get("predicted_winner") == truth[dk])
            rep[0] += ok; rep[1] += 1
            by_model[d["llm_model"]][0] += ok; by_model[d["llm_model"]][1] += 1
            docket_preds[dk][d["llm_model"]].append(d.get("predicted_winner"))

    # docket consensus across all replicates/models (modal)
    cons = [0, 0]
    for dk in truth:
        preds = [w for m in docket_preds.get(dk, {}).values() for w in m]
        if not preds:
            continue
        modal = modal_winner(preds)
        cons[0] += int(modal == truth[dk]); cons[1] += 1

    # majority-class baseline on the scored subset (always predict Petitioner=Reverse).
    # Restricted to dockets we actually forecast, so the baseline and the model
    # accuracies below are computed on the same cases.
    scored = [dk for dk in truth if docket_preds.get(dk)]
    pet = sum(1 for dk in scored if truth[dk] == "Petitioner")
    base = max(pet, len(scored) - pet) / len(scored)

    print(f"Decided OT2025-26 cases scored: {cons[1]} (of 62 forecast)")
    print(f"  Majority-class baseline on subset: {100*base:.1f}%  "
          f"({pet} Petitioner / {len(scored)-pet} Respondent)")
    p, lo, hi = wilson(rep[0], rep[1])
    print(f"  Replicate-level case accuracy: {p:.1f}%  [{lo:.1f}, {hi:.1f}]  (n={rep[1]})")
    p, lo, hi = wilson(cons[0], cons[1])
    print(f"  Docket-consensus accuracy:     {p:.1f}%  [{lo:.1f}, {hi:.1f}]  (n={cons[1]})")
    for m, (k, n) in sorted(by_model.items()):
        p, lo, hi = wilson(k, n)
        print(f"    {m}: {p:.1f}%  [{lo:.1f}, {hi:.1f}]  (n={n})")
    if fs_correct:
        kk = sum(fs_correct.values()); nn = len(fs_correct)
        p, lo, hi = wilson(kk, nn)
        print(f"  FantasySCOTUS crowd on same subset: {p:.1f}%  [{lo:.1f}, {hi:.1f}]  (n={nn})")


if __name__ == "__main__":
    main()
