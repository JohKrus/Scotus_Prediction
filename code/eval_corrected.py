#!/usr/bin/env python3
"""Corrected evaluation: each case counted once, assigned to its GT term."""

import csv
import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from scotus_v2 import config, evaluate

gt_dir = config.data_dir() / "ground_truth"
docket_term = {}
for f in sorted(gt_dir.glob("*.csv")):
    with open(f) as fh:
        for row in csv.DictReader(fh):
            docket_term[row["Docket"]] = int(row["Term"])

gt = evaluate.load_ground_truth()

# Show distribution per folder
for folder in ["term_22", "term_23", "term_24"]:
    d = config.results_dir() / "predictions" / "scotus_v2_final" / folder
    files = sorted(d.glob("*.json"))
    by_gt = defaultdict(int)
    for f in files:
        dk = f.stem.replace("_predictions", "")
        if dk in gt:
            by_gt[docket_term.get(dk, 0)] += 1
    print(f"{folder}: {dict(sorted(by_gt.items()))}")

print()

# Corrected eval: each docket counted once
seen = set()
for gt_term in [2022, 2023, 2024]:
    c_ok = 0
    c_tot = 0
    j_ok = 0
    j_tot = 0
    n = 0
    for folder in ["term_22", "term_23", "term_24"]:
        d = config.results_dir() / "predictions" / "scotus_v2_final" / folder
        for f in sorted(d.glob("*.json")):
            dk = f.stem.replace("_predictions", "")
            if dk not in gt or dk in seen:
                continue
            if docket_term.get(dk) != gt_term:
                continue
            seen.add(dk)
            a = gt[dk]
            n += 1
            with open(f) as fh:
                preds = json.load(fh)
            for p in preds:
                c_tot += 1
                if p["predicted_winner"] == a["winner"]:
                    c_ok += 1
                for fn, ab in evaluate.JUSTICE_NAME_MAP.items():
                    jv = p.get("justice_votes", {})
                    if fn in jv and ab in a["justice_votes"]:
                        j_tot += 1
                        if jv[fn]["vote"] == a["justice_votes"][ab]:
                            j_ok += 1
    if c_tot:
        print(f"GT Term {gt_term}: {n} cases | Case {c_ok}/{c_tot} ({c_ok/c_tot*100:.1f}%) | Justice {j_ok}/{j_tot} ({j_ok/j_tot*100:.1f}%)")

print(f"\nTotal unique cases evaluated: {len(seen)}")
