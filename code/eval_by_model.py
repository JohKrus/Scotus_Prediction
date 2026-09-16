#!/usr/bin/env python3
"""Corrected evaluation by GT term AND model."""

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

# folder -> model label
folder_model = {
    "term_22": "GPT-5.2",
    "term_23": "GPT-5.2",
    "term_24": "GPT-5.2",
    "term_22_claude": "Claude-4.6",
    "term_23_claude": "Claude-4.6",
    "term_24_claude": "Claude-4.6",
}

# Collect: (gt_term, model) -> results
results = defaultdict(lambda: {"c_ok": 0, "c_tot": 0, "j_ok": 0, "j_tot": 0, "n": 0})

for folder, model in folder_model.items():
    d = config.results_dir() / "predictions" / "scotus_v2_final" / folder
    if not d.exists():
        continue
    for f in sorted(d.glob("*.json")):
        dk = f.stem.replace("_predictions", "")
        if dk not in gt:
            continue
        gt_t = docket_term.get(dk)
        if gt_t is None:
            continue
        a = gt[dk]
        key = (gt_t, model)
        with open(f) as fh:
            preds = json.load(fh)
        for p in preds:
            results[key]["c_tot"] += 1
            if p["predicted_winner"] == a["winner"]:
                results[key]["c_ok"] += 1
            for fn, ab in evaluate.JUSTICE_NAME_MAP.items():
                jv = p.get("justice_votes", {})
                if fn in jv and ab in a["justice_votes"]:
                    results[key]["j_tot"] += 1
                    if jv[fn]["vote"] == a["justice_votes"][ab]:
                        results[key]["j_ok"] += 1
        results[key]["n"] += 1

# Print
models_seen = sorted(set(m for _, m in results.keys()))
terms_seen = sorted(set(t for t, _ in results.keys()))

print(f"{'Term':>6}", end="")
for m in models_seen:
    print(f"  | {m:>20} Case  Justice", end="")
print()
print("-" * (6 + len(models_seen) * 35))

for t in terms_seen:
    print(f"{t:>6}", end="")
    for m in models_seen:
        r = results.get((t, m))
        if r and r["c_tot"]:
            cpct = r["c_ok"] / r["c_tot"] * 100
            jpct = r["j_ok"] / r["j_tot"] * 100
            print(f"  | {r['n']:>3} cases  {cpct:>5.1f}%  {jpct:>5.1f}%", end="")
        else:
            print(f"  | {'—':>20}{'':>13}", end="")
    print()
