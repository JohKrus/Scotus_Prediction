"""Persona-ablation diagnostics for the deliberation pipeline.

A full ablation (re-running the pipeline with neutral, lean-free personas)
requires live API calls and is implemented separately in
run_ablation_neutral.py. This script computes a data-only PROXY from the
existing predictions + SCDB that bounds how much the persona priors can be
contributing:

  (1) Constant per-justice prior baseline: for each justice, the accuracy of
      always predicting that justice's modal vote (no case reading). If the
      model's justice-level accuracy barely exceeds this, then case-specific
      reasoning adds little over a fixed per-justice prior (which the persona
      encodes a version of).
  (2) Lean alignment: correlation between each justice's encoded
      conservative_lean and (a) their ACTUAL petitioner-rate and (b) the
      model's PREDICTED petitioner-rate. If the model's predicted rates track
      lean far more than actual rates do, the persona is injecting ideological
      signal that is not predictive of the petitioner/respondent target.

Run from repo root:  python v2/code/persona_ablation.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_predictions import load_scdb, JUSTICE_MAP, PRED_GLOB  # noqa: E402

# conservative_lean from scotus_v2/deliberation.py
LEAN = {
    "John Roberts": 0.65, "Clarence Thomas": 0.90, "Samuel Alito": 0.85,
    "Sonia Sotomayor": 0.15, "Elena Kagan": 0.35, "Neil Gorsuch": 0.75,
    "Brett Kavanaugh": 0.70, "Amy Coney Barrett": 0.80, "Ketanji Brown Jackson": 0.20,
}


def main():
    case_winner, justice_vote, _ = load_scdb()
    case_term = load_scdb.case_term

    SCOREABLE = {2022, 2023, 2024}
    CODES = set(JUSTICE_MAP.values())
    # actual votes per justice, restricted to scoreable Terms and the 9 sitting justices
    actual = defaultdict(lambda: [0, 0])     # justiceCode -> [petitioner, total]
    for (docket, code), v in justice_vote.items():
        if code not in CODES or case_term.get(docket) not in SCOREABLE:
            continue
        actual[code][1] += 1
        actual[code][0] += int(v == "Petitioner")

    # model predicted votes + correctness per justice
    pred = defaultdict(lambda: [0, 0])       # full name -> [pred_petitioner, total]
    correct = defaultdict(lambda: [0, 0])    # full name -> [correct, total]
    for f in glob.glob(PRED_GLOB):
        try:
            arr = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        for d in arr:
            docket = d.get("docket")
            if case_winner.get(docket) is None or case_term.get(docket) is None:
                continue
            for jname, jv in d.get("justice_votes", {}).items():
                code = JUSTICE_MAP.get(jname)
                tv = justice_vote.get((docket, code))
                if tv is None:
                    continue
                pv = jv.get("vote")
                pred[jname][1] += 1
                pred[jname][0] += int(pv == "Petitioner")
                correct[jname][1] += 1
                correct[jname][0] += int(pv == tv)

    code2name = {v: k for k, v in JUSTICE_MAP.items()}

    print("=" * 84)
    print("(1) CONSTANT PER-JUSTICE PRIOR vs MODEL  (pooled scoreable Terms)")
    print("=" * 84)
    print(f"  {'Justice':22s} {'actual Pet%':>11s} {'prior-only acc':>15s} {'model acc':>11s} {'gain':>7s}")
    prior_accs, model_accs = [], []
    for code, (kp, n) in sorted(actual.items()):
        name = code2name.get(code, code)
        pet_rate = kp / n
        prior_acc = max(pet_rate, 1 - pet_rate)        # always-modal-vote accuracy
        ck, cn = correct.get(name, [0, 0])
        macc = ck / cn if cn else float("nan")
        prior_accs.append(prior_acc); model_accs.append(macc)
        print(f"  {name:22s} {100*pet_rate:10.1f}% {100*prior_acc:14.1f}% "
              f"{100*macc:10.1f}% {100*(macc-prior_acc):+6.1f}")
    print(f"\n  mean prior-only accuracy: {100*np.mean(prior_accs):.1f}%")
    print(f"  mean model accuracy:      {100*np.mean(model_accs):.1f}%")
    print(f"  mean gain from case reading over per-justice prior: "
          f"{100*np.mean(np.array(model_accs)-np.array(prior_accs)):+.1f} pts")

    print("\n" + "=" * 84)
    print("(2) LEAN ALIGNMENT  (does the persona lean drive predictions more than reality?)")
    print("=" * 84)
    leans, actual_pet, pred_pet = [], [], []
    for name, lean in LEAN.items():
        code = JUSTICE_MAP[name]
        ka, na = actual.get(code, [0, 0])
        kp, npd = pred.get(name, [0, 0])
        if na and npd:
            leans.append(lean)
            actual_pet.append(ka / na)
            pred_pet.append(kp / npd)
            print(f"  {name:22s} lean={lean:.2f}  actual Pet%={100*ka/na:5.1f}  "
                  f"predicted Pet%={100*kp/npd:5.1f}")
    leans = np.array(leans); actual_pet = np.array(actual_pet); pred_pet = np.array(pred_pet)
    print(f"\n  corr(lean, ACTUAL petitioner-rate):    {np.corrcoef(leans, actual_pet)[0,1]:+.3f}")
    print(f"  corr(lean, PREDICTED petitioner-rate): {np.corrcoef(leans, pred_pet)[0,1]:+.3f}")
    print("  (Petitioner/respondent is not an ideological axis, so both should be near zero;"
          "\n   a large PREDICTED correlation would indicate the persona prior injects"
          "\n   spurious ideological signal. Compare the two.)")


if __name__ == "__main__":
    main()
