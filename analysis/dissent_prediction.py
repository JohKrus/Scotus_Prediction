"""Does the pipeline know WHO dissents? Scored from the saved prediction artifacts.

A justice's predicted dissent is a final vote against the run's own predicted
winner; the actual dissent is a vote against the Court's winner (SCDB `majority`
for OT2022-24; the corrected OT2025-26 results sheet otherwise). No API calls.

Four questions, each against a baseline that needs no case reading:
  1. Divided or unanimous?   vs. the Term's majority class (divided or unanimous).
  2. Dissent classification  (precision / recall / F1 per justice-vote).
  3. Ranking: does a dissent score order justices by dissent risk?  AUC vs. each
     justice's dissent rate over the three prior Terms (SCDB, forward-chained).
  4. Who, given how many: in actually divided cases, take the k justices with the
     highest dissent score (k = actual number of dissenters) and measure overlap
     with the real dissenters, vs. the same top-k by prior-Term dissent rate. Both
     are told k, so the comparison isolates "who" from "how many".

Run from repo root:  python v2/code/dissent_prediction.py
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys
from collections import defaultdict

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from paths import DATA_DIR  # noqa: E402

from parse_justice_votes_ot2526 import PERSONA_TO_SURNAME, load_justice_truth  # noqa: E402
from score_predictions import (JUSTICE_CSV, JUSTICE_MAP, PRED_GLOB, # noqa: E402
                               load_scdb)

OUT_CSV = os.path.join(HERE, "cache", "dissent_prediction_summary.csv")
TERMS = (2022, 2023, 2024, 2025)
SURNAME = {code: PERSONA_TO_SURNAME[name] for name, code in JUSTICE_MAP.items()}


# ----------------------------------------------------------------------------- truth
def load_truth():
    """(term_of, winner, dissent); dissent[(docket, surname)] = bool, recusals omitted."""
    case_winner, _, _ = load_scdb()
    term_of = {d: t for d, t in load_scdb.case_term.items() if t in (2022, 2023, 2024)}

    jc = pd.read_csv(JUSTICE_CSV, encoding="latin-1", low_memory=False,
                     usecols=["term", "docket", "justiceName", "majority"])
    dissent = {}
    for r in jc.itertuples(index=False):
        d = str(r.docket)
        if d in term_of and r.justiceName in SURNAME and r.majority in (1.0, 2.0):
            dissent[(d, SURNAME[r.justiceName])] = r.majority == 1.0

    x = pd.read_excel(os.path.join(DATA_DIR, "scotus_ot2025_cases.xlsx"))
    winner = {d: w for d, w in case_winner.items() if d in term_of}
    sheet = dict(zip(x["Docket number"].astype(str).str.strip(),
                     x["Winner (petitioner/respondent)"]))
    jtruth, _ = load_justice_truth()
    for d, votes in jtruth.items():
        term_of[d] = 2025
        winner[d] = sheet[d]
        for sur, v in votes.items():
            dissent[(d, sur)] = v != sheet[d]
    return term_of, winner, dissent


def prior_dissent_rates() -> dict[int, dict[str, float]]:
    """Each justice's dissent rate over the three Terms before each test Term (SCDB)."""
    jc = pd.read_csv(JUSTICE_CSV, encoding="latin-1", low_memory=False,
                     usecols=["term", "docket", "justiceName", "majority"])
    jc = jc[jc.majority.isin([1.0, 2.0]) & jc.justiceName.isin(SURNAME)]
    jc = jc.drop_duplicates(["docket", "justiceName"])
    rates = {}
    for t in TERMS:
        sub = jc[jc.term.between(t - 3, t - 1)]
        r = sub.groupby("justiceName").majority.apply(lambda m: (m == 1.0).mean())
        pooled = (sub.majority == 1.0).mean()
        # A justice with no prior Terms (Jackson before OT2023) gets the pooled rate.
        rates[t] = {s: float(r.get(code, pooled)) for code, s in SURNAME.items()}
    return rates


# ----------------------------------------------------------------------------- predictions
def load_runs(term_of, winner):
    """One record per run: term, model, docket, dissent flags and scores per surname."""
    runs = []
    for f in glob.glob(PRED_GLOB):
        for p in json.load(open(f, encoding="utf-8")):
            d = p.get("docket")
            if d not in term_of:
                continue
            win = p.get("predicted_winner")
            flags, scores = {}, {}
            for persona, jv in p.get("justice_votes", {}).items():
                sur = PERSONA_TO_SURNAME.get(persona)
                if sur is None:
                    continue
                against = jv.get("vote") != win
                conf = jv.get("confidence") or 0.5
                flags[sur] = against
                # Dissent score: stated confidence on the minority side, else its complement.
                scores[sur] = conf if against else 1 - conf
            runs.append({"term": term_of[d], "model": p.get("llm_model"), "docket": d,
                         "win_ok": win == winner[d], "flags": flags, "scores": scores})
    return runs


# ----------------------------------------------------------------------------- metrics
def auc(pos: list[float], neg: list[float]) -> float:
    """Mann-Whitney AUC with ties counted as half."""
    if not pos or not neg:
        return float("nan")
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    rank_sum, i = 0.0, 0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        avg = (i + 1 + j) / 2
        rank_sum += avg * sum(lbl for _, lbl in allv[i:j])
        i = j
    n1, n0 = len(pos), len(neg)
    return (rank_sum - n1 * (n1 + 1) / 2) / (n1 * n0)


def topk_overlap(score: dict[str, float], actual: set[str]) -> float:
    k = len(actual)
    top = sorted(score, key=lambda s: (-score[s], s))[:k]
    return len(set(top) & actual) / k


def evaluate(runs, dissent, rates):
    rows = []
    for (t, m) in sorted({(r["term"], r["model"]) for r in runs}):
        rr = [r for r in runs if r["term"] == t and r["model"] == m]
        tp = fp = fn = tn = 0
        unan_ok = unan_n = 0
        pos_s, neg_s, pos_b, neg_b = [], [], [], []
        ov_model, ov_base, exact = [], [], []
        # Same classification restricted to runs with the right winner: there the
        # predicted and actual minorities sit on the same side of the case.
        ctp = cfp = cfn = 0
        n_pred_div = n_true_div = 0
        for r in rr:
            truth = {s: dissent[(r["docket"], s)] for s in r["flags"] if (r["docket"], s) in dissent}
            if not truth:
                continue
            actual = {s for s, v in truth.items() if v}
            pred = {s for s in truth if r["flags"][s]}
            for s, v in truth.items():
                f = r["flags"][s]
                tp += f and v; fp += f and not v; fn += (not f) and v; tn += (not f) and not v
                if r["win_ok"]:
                    ctp += f and v; cfp += f and not v; cfn += (not f) and v
                (pos_s if v else neg_s).append(r["scores"][s])
                (pos_b if v else neg_b).append(rates[t][s])
            unan_ok += (not pred) == (not actual); unan_n += 1
            n_pred_div += bool(pred); n_true_div += bool(actual)
            if actual:
                ov_model.append(topk_overlap({s: r["scores"][s] for s in truth}, actual))
                ov_base.append(topk_overlap({s: rates[t][s] for s in truth}, actual))
                exact.append(pred == actual)
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if tp else 0.0
        cprec = ctp / (ctp + cfp) if ctp + cfp else float("nan")
        crec = ctp / (ctp + cfn) if ctp + cfn else float("nan")
        rows.append({
            "term": t, "model": m, "runs": unan_n,
            "true_divided_pct": 100 * n_true_div / unan_n,
            "pred_divided_pct": 100 * n_pred_div / unan_n,
            "unanimity_acc": 100 * unan_ok / unan_n,
            "always_unanimous_acc": 100 * (unan_n - n_true_div) / unan_n,
            "dissent_rate": 100 * (tp + fn) / (tp + fp + fn + tn),
            "precision": 100 * prec, "recall": 100 * rec, "f1": 100 * f1,
            "precision_right_winner": 100 * cprec, "recall_right_winner": 100 * crec,
            "auc_model": auc(pos_s, neg_s), "auc_prior_rate": auc(pos_b, neg_b),
            "divided_runs": len(ov_model),
            "who_topk_model": 100 * sum(ov_model) / len(ov_model) if ov_model else float("nan"),
            "who_topk_prior": 100 * sum(ov_base) / len(ov_base) if ov_base else float("nan"),
            "exact_set_pct": 100 * sum(exact) / len(exact) if exact else float("nan"),
        })
    return rows


def bootstrap_contrasts(runs, dissent, rates, models=("Claude-4.6", "GPT-5.2"),
                        reps=2000, seed=7):
    """Per Term, Claude and GPT pooled: model-minus-baseline contrasts with 95% intervals
    from a case-level (docket) bootstrap, since a case's runs and justices are not independent."""
    import random
    rng = random.Random(seed)
    out = []
    for t in TERMS:
        by_d = defaultdict(lambda: {"unan": [], "pm": [], "pb": [], "tm": [], "tb": []})
        for r in runs:
            if r["term"] != t or r["model"] not in models:
                continue
            truth = {s: dissent[(r["docket"], s)] for s in r["flags"] if (r["docket"], s) in dissent}
            if not truth:
                continue
            actual = {s for s, v in truth.items() if v}
            pred = {s for s in truth if r["flags"][s]}
            b = by_d[r["docket"]]
            b["unan"].append(((not pred) == (not actual), bool(actual)))
            b["pm"] += [(r["scores"][s], v) for s, v in truth.items()]
            b["pb"] += [(rates[t][s], v) for s, v in truth.items()]
            if actual:
                b["tm"].append(topk_overlap({s: r["scores"][s] for s in truth}, actual))
                b["tb"].append(topk_overlap({s: rates[t][s] for s in truth}, actual))
        dockets = sorted(by_d)

        # The baseline's class (divided or unanimous) is fixed on the full Term; re-picking
        # it per resample would take the max of two near-50% shares and bias it upward.
        all_unan = [u for d in dockets for u in by_d[d]["unan"]]
        base_divided = 2 * sum(dv for _, dv in all_unan) >= len(all_unan)

        def stats(sample):
            unan = [u for d in sample for u in by_d[d]["unan"]]
            acc = sum(ok for ok, _ in unan) / len(unan)
            div = sum(dv for _, dv in unan) / len(unan)
            pm = [p for d in sample for p in by_d[d]["pm"]]
            pb = [p for d in sample for p in by_d[d]["pb"]]
            tm = [x for d in sample for x in by_d[d]["tm"]]
            tb = [x for d in sample for x in by_d[d]["tb"]]
            a_m = auc([s for s, v in pm if v], [s for s, v in pm if not v])
            a_b = auc([s for s, v in pb if v], [s for s, v in pb if not v])
            return (100 * (acc - (div if base_divided else 1 - div)), a_m - a_b,
                    100 * (sum(tm) / len(tm) - sum(tb) / len(tb)) if tm else float("nan"))

        point = stats(dockets)
        draws = [stats([rng.choice(dockets) for _ in dockets]) for _ in range(reps)]
        ci = []
        for i in range(3):
            v = sorted(x[i] for x in draws if x[i] == x[i])
            ci.append((v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]))
        out.append((t, len(dockets), point, ci))
    return out


def main():
    term_of, winner, dissent = load_truth()
    rates = prior_dissent_rates()
    runs = load_runs(term_of, winner)
    rows = evaluate(runs, dissent, rates)

    print("Prior-Term dissent rates used as the no-reading baseline (%):")
    for t in TERMS:
        print(f"  OT{t}: " + "  ".join(f"{s[:4]} {100*v:4.1f}" for s, v in rates[t].items()))

    print("\n1-2. DIVIDED vs UNANIMOUS, and DISSENT CLASSIFICATION (per justice-vote)")
    print(f"  {'term':6s} {'model':11s} {'runs':>4s} {'div%':>5s} {'pdiv%':>5s} "
          f"{'unan':>5s} {'base':>5s} | {'diss%':>5s} {'prec':>5s} {'rec':>5s} {'F1':>5s} | "
          f"{'precW':>5s} {'recW':>5s}")
    for r in rows:
        print(f"  OT{r['term']} {r['model']:11s} {r['runs']:4d} {r['true_divided_pct']:5.1f} "
              f"{r['pred_divided_pct']:5.1f} {r['unanimity_acc']:5.1f} {r['always_unanimous_acc']:5.1f} | "
              f"{r['dissent_rate']:5.1f} {r['precision']:5.1f} {r['recall']:5.1f} {r['f1']:5.1f} | "
              f"{r['precision_right_winner']:5.1f} {r['recall_right_winner']:5.1f}")
    print("  div% = actually divided; pdiv% = predicted divided; unan = accuracy on divided-vs-unanimous;"
          "\n  base = always-unanimous accuracy; diss% = share of justice-votes that are dissents;"
          "\n  precW / recW = precision / recall on runs that got the winner right.")

    print("\n3-4. RANKING and WHO-DISSENTS-GIVEN-HOW-MANY")
    print(f"  {'term':6s} {'model':11s} {'AUC':>5s} {'AUCp':>5s} | {'nDiv':>4s} "
          f"{'topk':>5s} {'topkP':>5s} {'exact':>5s}")
    for r in rows:
        print(f"  OT{r['term']} {r['model']:11s} {r['auc_model']:5.3f} {r['auc_prior_rate']:5.3f} | "
              f"{r['divided_runs']:4d} {r['who_topk_model']:5.1f} {r['who_topk_prior']:5.1f} "
              f"{r['exact_set_pct']:5.1f}")
    print("  AUC = dissent score vs. actual dissent; AUCp = same with prior-Term dissent rate;"
          "\n  topk / topkP = % of actual dissenters found among the k highest-scored justices"
          "\n  (model / prior rate), divided cases only; exact = predicted dissenter set is exactly right.")

    print("\n5. CONTRASTS vs. NO-READING BASELINES (Claude + GPT pooled; 95% case-bootstrap CI)")
    print(f"  {'term':6s} {'cases':>5s}  {'divided-vs-unanimous acc - majority class':>42s}"
          f"  {'AUC - prior-rate AUC':>22s}  {'top-k - prior top-k':>22s}")
    for t, n, pt, ci in bootstrap_contrasts(runs, dissent, rates):
        print(f"  OT{t} {n:5d}  {pt[0]:+14.1f} pts [{ci[0][0]:+5.1f}, {ci[0][1]:+5.1f}]        "
              f"{pt[1]:+.3f} [{ci[1][0]:+.3f}, {ci[1][1]:+.3f}]  "
              f"{pt[2]:+5.1f} [{ci[2][0]:+5.1f}, {ci[2][1]:+5.1f}]")
    print("  Majority class = always predicting whichever of divided/unanimous was more common"
          "\n  in that Term (an in-sample, hence generous, baseline).")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
