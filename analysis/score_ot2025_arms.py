"""Score the September 2026 GPT-5.2 re-runs of OT2025-26 against the Court's record.

Arms (v2/predictions/ot2025_reruns/<arm>/<docket>_predictions.json, copied from
the run machine):
  control      v2, filings only       -- noise floor vs. the March pre-registered
                                         GPT-5.2 runs, and the no-transcript arm
  transcripts  v2, filings + oral-argument transcript
  v3           v3 (justice profiles from each justice's own opinions), filings only

All comparisons are paired on the same decided cases. Per case, "accuracy" is the
share of the two replicates that got the winner right; the paired difference gets
a case-bootstrap 95% interval, and the case-level majority call (ties to the
petitioner) gets an exact McNemar test.

Run from repo root:  python v2/code/score_ot2025_arms.py
"""
from __future__ import annotations

import glob
import json
import math
import os
import random
import sys
from collections import defaultdict

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from paths import DATA_DIR, PRED_DIR  # noqa: E402

from parse_justice_votes_ot2526 import PERSONA_TO_SURNAME, load_justice_truth  # noqa: E402
from score_predictions import modal_winner, wilson  # noqa: E402
from dissent_prediction import auc  # noqa: E402

RERUNS = os.path.join(PRED_DIR, "ot2025_reruns")
MARCH = os.path.join(PRED_DIR, "term_2025_2026", "*_gpt52_predictions.json")
ARMS = ("control", "transcripts", "v3")


def load_truth() -> dict[str, str]:
    x = pd.read_excel(os.path.join(DATA_DIR, "scotus_ot2025_cases.xlsx"))
    return {str(d).strip(): w for d, w in zip(x["Docket number"], x["Winner (petitioner/respondent)"])
            if isinstance(w, str)}


def load_arm(arm: str) -> dict[str, list[dict]]:
    pattern = MARCH if arm == "march" else os.path.join(RERUNS, arm, "*_predictions.json")
    out = {}
    for f in glob.glob(pattern):
        runs = [r for r in json.load(open(f, encoding="utf-8")) if r.get("llm_model") == "GPT-5.2"]
        if runs:
            out[runs[0]["docket"]] = runs
    return out


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def paired(a: dict[str, float], b: dict[str, float], reps=4000, seed=11):
    keys = sorted(set(a) & set(b))
    rng = random.Random(seed)
    diff = [a[k] - b[k] for k in keys]
    est = sum(diff) / len(diff)
    draws = sorted(sum(diff[rng.randrange(len(keys))] for _ in keys) / len(keys) for _ in range(reps))
    return est, draws[int(0.025 * reps)], draws[int(0.975 * reps) - 1], len(keys)


def summarize(name, runs_by_docket, truth, jtruth):
    acc, case_call, jk, jn = {}, {}, 0, 0
    scores, labels = [], []
    for d, runs in runs_by_docket.items():
        if d not in truth:
            continue
        acc[d] = sum(r["predicted_winner"] == truth[d] for r in runs) / len(runs)
        case_call[d] = modal_winner([r["predicted_winner"] for r in runs])
        pet = sum(sum(v["vote"] == "Petitioner" for v in r["justice_votes"].values()) for r in runs) / len(runs)
        scores.append(pet); labels.append(truth[d] == "Petitioner")
        for r in runs:
            for persona, v in r["justice_votes"].items():
                t = jtruth.get(d, {}).get(PERSONA_TO_SURNAME.get(persona))
                if t is not None:
                    jk += v["vote"] == t; jn += 1
    k = sum(acc.values()) * 2; n = 2 * len(acc)
    p, lo, hi = wilson(round(k), n)
    base = 100 * sum(truth[d] == "Petitioner" for d in acc) / len(acc)
    a = auc([s for s, l in zip(scores, labels) if l], [s for s, l in zip(scores, labels) if not l])
    print(f"  {name:12s} cases {len(acc):3d} | per run {p:5.1f}% [{lo:4.1f}, {hi:4.1f}] | "
          f"case call {100*sum(case_call[d]==truth[d] for d in acc)/len(acc):5.1f}% | "
          f"justice {100*jk/max(jn,1):5.1f}% | AUC {a:.3f} | baseline {base:5.1f}%")
    return acc, case_call


def main():
    truth, (jtruth, _) = load_truth(), load_justice_truth()
    arms = {"march": load_arm("march")}
    for arm in ARMS:
        arms[arm] = load_arm(arm)
    missing = [a for a in ARMS if not arms[a]]
    if missing:
        print(f"no runs found yet for: {', '.join(missing)} (expected under {RERUNS})")

    print("ACCURACY ON DECIDED OT2025 CASES (GPT-5.2; 2 replicates per case)")
    res = {a: summarize(a, r, truth, jtruth) for a, r in arms.items() if r}

    print("\nPAIRED COMPARISONS (same cases; per-case accuracy difference, case-bootstrap 95% CI;"
          " exact McNemar on case calls)")
    pairs = [("control", "march", "noise floor: new control vs. March runs"),
             ("transcripts", "control", "effect of oral-argument transcripts"),
             ("v3", "control", "effect of v3 justice profiles")]
    for x, y, label in pairs:
        if x not in res or y not in res:
            continue
        (ax, cx), (ay, cy) = res[x], res[y]
        est, lo, hi, n = paired(ax, ay)
        common = set(cx) & set(cy)
        b = sum(cx[d] == truth[d] and cy[d] != truth[d] for d in common)
        c = sum(cx[d] != truth[d] and cy[d] == truth[d] for d in common)
        agree = 100 * sum(cx[d] == cy[d] for d in common) / len(common)
        print(f"  {label:42s} n={n:2d}  {100*est:+5.1f} pts [{100*lo:+5.1f}, {100*hi:+5.1f}]  "
              f"case calls agree {agree:4.1f}%  McNemar {b}:{c} p={mcnemar_exact(b, c):.2f}")


if __name__ == "__main__":
    main()
