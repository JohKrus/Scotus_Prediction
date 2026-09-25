"""Failure diagnosis and decision-rule search across OT2022-2025, for OT2026-27 pre-registration.

All four terms are used for fitting. Because that leaves no held-out term, every
rule with a free parameter is also scored under leave-one-term-out CV: the
parameter is chosen on three terms and tested on the fourth, rotating. The CV
number -- not the pooled fit -- is the honest expectation for OT2026-27.

Everything here re-aggregates the *saved* persona votes. Rules that need a
different pipeline (oral-argument transcripts, dropped deliberation) are
identified but must be re-run to be tested.

Run from repo root:  python v2/code/optimize_rules.py
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

from score_predictions import modal_winner, wilson  # noqa: E402

TERMS = {"2022_2023": 2022, "2023_2024": 2023, "2024_2025": 2024, "2025_2026": 2025}
ALL_TERMS = [2022, 2023, 2024, 2025]


# ---------------------------------------------------------------- data
def load_all() -> dict[int, list[dict]]:
    sc = json.load(open(os.path.join(HERE, "cache", "scdb_truth.json")))
    x = pd.read_excel(os.path.join(DATA_DIR, "scotus_ot2025_cases.xlsx"))
    ot25 = {str(r["Docket number"]).strip(): r["Winner (petitioner/respondent)"].strip()
            for _, r in x.iterrows()
            if isinstance(r["Winner (petitioner/respondent)"], str)}

    by_term: dict[int, list[dict]] = defaultdict(list)
    for term_key, term in TERMS.items():
        truth = ot25 if term == 2025 else sc["case_winner"]
        for f in glob.glob(os.path.join(PRED_DIR, f"term_{term_key}", "*.json")):
            try:
                arr = json.load(open(f, encoding="utf-8"))
            except Exception:
                continue
            for p in arr:
                t = truth.get(p["docket"])
                if t is None:
                    continue
                real_term = 2025 if term == 2025 else sc["case_term"].get(p["docket"])
                if real_term is None:
                    continue
                by_term[int(real_term)].append(dict(p, _truth=t, _term=int(real_term)))
    return by_term


def persona_votes(p: dict, key: str = "justice_votes") -> dict[str, str]:
    jv = p.get(key) or {}
    return {k: v.get("vote") for k, v in jv.items() if isinstance(v, dict)}


# ---------------------------------------------------------------- rules
def rule_current(p, **_):
    return p.get("predicted_winner")


def rule_initial(p, **_):
    return p.get("initial_winner")


def rule_always_pet(p, **_):
    return "Petitioner"


def rule_conf_weighted(p, **_):
    s = 0.0
    for v in (p.get("justice_votes") or {}).values():
        if not isinstance(v, dict):
            continue
        c = v.get("confidence")
        c = c if isinstance(c, (int, float)) else 0.5
        s += c if v.get("vote") == "Petitioner" else -c
    return "Petitioner" if s >= 0 else "Respondent"


def make_threshold(k: int):
    """Call Respondent only when >= k of the nine personas say Respondent."""
    def f(p, **_):
        v = persona_votes(p)
        if not v:
            return p.get("predicted_winner")
        resp = sum(x == "Respondent" for x in v.values())
        return "Respondent" if resp >= k else "Petitioner"
    f.__name__ = f"resp_ge_{k}"
    return f


def make_subset(names: tuple[str, ...]):
    def f(p, **_):
        v = persona_votes(p)
        sel = [v[n] for n in names if n in v]
        if not sel:
            return p.get("predicted_winner")
        pet = sum(x == "Petitioner" for x in sel)
        if pet * 2 == len(sel):
            return p.get("predicted_winner")
        return "Petitioner" if pet * 2 > len(sel) else "Respondent"
    f.__name__ = "subset"
    return f


SWING = ("John Roberts", "Brett Kavanaugh", "Amy Coney Barrett")


# ---------------------------------------------------------------- scoring
def score(rule, recs) -> tuple[int, int]:
    k = n = 0
    for p in recs:
        pred = rule(p)
        if pred is None:
            continue
        k += int(pred == p["_truth"]); n += 1
    return k, n


def consensus(rule, recs) -> tuple[int, int]:
    by_d = defaultdict(list)
    for p in recs:
        pred = rule(p)
        if pred is not None:
            by_d[p["docket"]].append((pred, p["_truth"]))
    k = n = 0
    for vals in by_d.values():
        preds = [v[0] for v in vals]
        k += int(modal_winner(preds) == vals[0][1]); n += 1
    return k, n


def line(label, k, n, ref=None):
    p, lo, hi = wilson(k, n)
    d = f"  {p-ref:+5.1f}" if ref is not None else ""
    return f"  {label:<32s} {p:5.1f}%  [{lo:4.1f}, {hi:4.1f}]  (n={n:4d}){d}"


def hdr(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)


# ---------------------------------------------------------------- main
def main() -> None:
    by_term = load_all()
    allrecs = [p for t in ALL_TERMS for p in by_term.get(t, [])]

    hdr("A. RELIABILITY — is the signal stable across identical re-runs?")
    # same docket, same model, 2 replicates: do they agree?
    pair = defaultdict(list)
    for p in allrecs:
        pair[(p["_term"], p["docket"], p["llm_model"])].append(p)
    agree = tot = 0
    jagree = jtot = 0
    for recs in pair.values():
        if len(recs) < 2:
            continue
        a, b = recs[0], recs[1]
        agree += int(a["predicted_winner"] == b["predicted_winner"]); tot += 1
        va, vb = persona_votes(a), persona_votes(b)
        for name in set(va) & set(vb):
            jagree += int(va[name] == vb[name]); jtot += 1
    print(f"  Case-call agreement between the 2 replicates: {100*agree/tot:.1f}%  (n={tot} pairs)")
    print(f"  Persona-vote agreement between replicates:    {100*jagree/jtot:.1f}%  (n={jtot} votes)")
    print("  (temperature 0.3; disagreement here is pure sampling noise, not signal)")

    hdr("B. DISCRIMINATION — does the model vary its call across cases?")
    for t in ALL_TERMS:
        recs = by_term.get(t, [])
        if not recs:
            continue
        by_d = defaultdict(list)
        for p in recs:
            by_d[p["docket"]].append(p["predicted_winner"] == "Petitioner")
        rates = [sum(v) / len(v) for v in by_d.values()]
        # share of dockets where every run agreed
        firm = sum(1 for r in rates if r in (0.0, 1.0)) / len(rates)
        print(f"  OT{t}: mean petitioner rate {100*sum(rates)/len(rates):5.1f}%  |  "
              f"dockets with unanimous runs {100*firm:5.1f}%")

    hdr("C. RULE SEARCH — pooled fit on all four terms")
    base_k, base_n = score(rule_always_pet, allrecs)
    base = 100 * base_k / base_n
    print(f"  majority-class baseline (all terms pooled): {base:.1f}%\n")
    fixed = [
        ("current (deliberated)", rule_current),
        ("no deliberation (initial)", rule_initial),
        ("confidence-weighted", rule_conf_weighted),
        ("swing three only", make_subset(SWING)),
        ("always petitioner", rule_always_pet),
    ]
    for lab, r in fixed:
        k, n = score(r, allrecs)
        print(line(lab, k, n, base))
    print()
    for k_ in range(1, 10):
        kk, nn = score(make_threshold(k_), allrecs)
        print(line(f"respondent needs >= {k_} of 9", kk, nn, base))

    hdr("D. LEAVE-ONE-TERM-OUT CV — the honest expectation for OT2026-27")
    print("  Threshold k is chosen on three terms, tested on the held-out fourth.\n")
    tot_k = tot_n = 0
    base_tot_k = base_tot_n = 0
    for held in ALL_TERMS:
        fit = [p for t in ALL_TERMS if t != held for p in by_term.get(t, [])]
        test = by_term.get(held, [])
        if not test:
            continue
        best_k, best_acc = None, -1.0
        for k_ in range(1, 10):
            kk, nn = score(make_threshold(k_), fit)
            if nn and 100 * kk / nn > best_acc:
                best_acc, best_k = 100 * kk / nn, k_
        kk, nn = score(make_threshold(best_k), test)
        bk, bn = score(rule_always_pet, test)
        ck, cn = score(rule_current, test)
        tot_k += kk; tot_n += nn
        base_tot_k += bk; base_tot_n += bn
        print(f"  hold out OT{held}: fit picks k={best_k} -> test {100*kk/nn:5.1f}%   "
              f"| current rule {100*ck/cn:5.1f}%  | baseline {100*bk/bn:5.1f}%")
    p, lo, hi = wilson(tot_k, tot_n)
    print(f"\n  Pooled CV accuracy of the tuned rule: {p:.1f}%  [{lo:.1f}, {hi:.1f}]")
    p2, _, _ = wilson(base_tot_k, base_tot_n)
    print(f"  Pooled baseline on the same folds:    {p2:.1f}%")
    print(f"  ==> expected edge over baseline next term: {p-p2:+.1f} points")

    hdr("E. PER-TERM BREAKDOWN OF THE LEADING RULES")
    cands = [("current", rule_current), ("no deliberation", rule_initial),
             ("always petitioner", rule_always_pet)]
    for k_ in (4, 5):
        cands.append((f"resp>={k_}", make_threshold(k_)))
    print(f"  {'rule':<22s}" + "".join(f"  OT{t}" for t in ALL_TERMS))
    for lab, r in cands:
        cells = []
        for t in ALL_TERMS:
            kk, nn = score(r, by_term.get(t, []))
            cells.append(f"{100*kk/nn:5.1f}" if nn else "   --")
        print(f"  {lab:<22s}" + "".join(f"  {c}" for c in cells))

    hdr("F. SELECTIVE PREDICTION — is there a subset worth trusting?")
    for t in [2025]:
        recs = by_term.get(t, [])
        by_d = defaultdict(list)
        for p in recs:
            by_d[p["docket"]].append(p)
        buckets = defaultdict(lambda: [0, 0])
        for d, rr in by_d.items():
            preds = [x["predicted_winner"] for x in rr]
            n_agree = max(preds.count("Petitioner"), preds.count("Respondent"))
            modal = modal_winner(preds)
            buckets[n_agree][0] += int(modal == rr[0]["_truth"])
            buckets[n_agree][1] += 1
        print(f"  OT{t}, by how many of the 4 runs agreed:")
        for nn_ in sorted(buckets, reverse=True):
            kk, tot_ = buckets[nn_]
            print(f"    {nn_}/4 agree: {100*kk/tot_:5.1f}%  (n={tot_} dockets)")


if __name__ == "__main__":
    main()
