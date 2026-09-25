"""Alternative decision rules for the v2 pipeline, fit on OT2022-24, tested on OT2025-26.

Everything here is a re-aggregation of the *saved* persona votes: no new API calls
and no re-running of the deliberation graph. The point is to separate two failure
modes -- a bad aggregation rule sitting on top of a usable signal, versus a signal
that is simply too weak to aggregate.

Protocol: any rule with a free parameter (a vote threshold, a confidence cutoff,
a persona subset) has that parameter chosen on the OT2022-24 backtest terms only,
then applied unchanged to OT2025-26. Rules tuned directly on OT2025-26 are
reported separately and labelled as such -- they are an upper bound on what
hindsight buys, not a forecast.

Run from repo root:  python v2/code/optimize_ot2526.py
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
FIT_TERMS = [2022, 2023, 2024]
TEST_TERM = 2025


# ---------------------------------------------------------------- data
def load_all() -> dict[int, list[dict]]:
    sc = json.load(open(os.path.join(HERE, "cache", "scdb_truth.json")))
    x = pd.read_excel(os.path.join(DATA_DIR, "scotus_ot2025_cases.xlsx"))
    ot25 = {str(r["Docket number"]).strip(): r["Winner (petitioner/respondent)"].strip()
            for _, r in x.iterrows()
            if isinstance(r["Winner (petitioner/respondent)"], str)}

    by_term: dict[int, list[dict]] = defaultdict(list)
    for term_key, term in TERMS.items():
        truth = ot25 if term == TEST_TERM else sc["case_winner"]
        for f in glob.glob(os.path.join(PRED_DIR, f"term_{term_key}", "*.json")):
            try:
                arr = json.load(open(f, encoding="utf-8"))
            except Exception:
                continue
            for p in arr:
                t = truth.get(p["docket"])
                if t is None:
                    continue
                # group by the term the case actually belongs to for backtests
                real_term = term if term == TEST_TERM else sc["case_term"].get(p["docket"])
                if real_term is None:
                    continue
                p = dict(p, _truth=t, _term=int(real_term))
                by_term[int(real_term)].append(p)
    return by_term


def persona_counts(p: dict, key: str = "justice_votes") -> tuple[int, int]:
    """(petitioner_votes, total) among the nine personas."""
    jv = p.get(key) or {}
    votes = [v.get("vote") for v in jv.values() if isinstance(v, dict)]
    return sum(v == "Petitioner" for v in votes), len(votes)


# ---------------------------------------------------------------- rules
# A rule maps one prediction record -> "Petitioner" / "Respondent".

def rule_current(p, **_):
    return p.get("predicted_winner")


def rule_no_deliberation(p, **_):
    return p.get("initial_winner")


def rule_always_petitioner(p, **_):
    return "Petitioner"


def rule_persona_majority(p, **_):
    pet, n = persona_counts(p)
    if n == 0:
        return p.get("predicted_winner")
    return "Petitioner" if pet * 2 > n else "Respondent"


def rule_conf_weighted(p, **_):
    jv = p.get("justice_votes") or {}
    s = 0.0
    for v in jv.values():
        if not isinstance(v, dict):
            continue
        c = v.get("confidence")
        c = c if isinstance(c, (int, float)) else 0.5
        s += c if v.get("vote") == "Petitioner" else -c
    return "Petitioner" if s >= 0 else "Respondent"


def make_threshold_rule(k: int):
    """Predict Respondent only when at least k of 9 personas say Respondent."""
    def f(p, **_):
        pet, n = persona_counts(p)
        if n == 0:
            return p.get("predicted_winner")
        return "Respondent" if (n - pet) >= k else "Petitioner"
    f.__name__ = f"respondent_needs_{k}_of_9"
    return f


def make_subset_rule(names: tuple[str, ...]):
    """Median-voter style: aggregate only the named personas."""
    def f(p, **_):
        jv = p.get("justice_votes") or {}
        votes = [v.get("vote") for k, v in jv.items() if k in names and isinstance(v, dict)]
        if not votes:
            return p.get("predicted_winner")
        pet = sum(v == "Petitioner" for v in votes)
        if pet * 2 == len(votes):
            return p.get("predicted_winner")   # tie -> fall back
        return "Petitioner" if pet * 2 > len(votes) else "Respondent"
    f.__name__ = "subset_" + "_".join(n.split()[-1] for n in names)
    return f


SWING = ("John Roberts", "Brett Kavanaugh", "Amy Coney Barrett", "Neil Gorsuch")


# ---------------------------------------------------------------- eval
def score(rule, recs, **kw) -> tuple[int, int]:
    k = n = 0
    for p in recs:
        pred = rule(p, **kw)
        if pred is None:
            continue
        k += int(pred == p["_truth"]); n += 1
    return k, n


def consensus_score(rule, recs, **kw) -> tuple[int, int]:
    """Docket-level modal prediction across all replicates/models."""
    by_docket = defaultdict(list)
    for p in recs:
        pred = rule(p, **kw)
        if pred is not None:
            by_docket[p["docket"]].append((pred, p["_truth"]))
    k = n = 0
    for _, vals in by_docket.items():
        preds = [v[0] for v in vals]
        modal = modal_winner(preds)
        k += int(modal == vals[0][1]); n += 1
    return k, n


def line(label, k, n, ref=None):
    p, lo, hi = wilson(k, n)
    delta = f"  {p-ref:+5.1f}" if ref is not None else ""
    return f"  {label:<34s} {p:5.1f}%  [{lo:4.1f}, {hi:4.1f}]  (n={n:4d}){delta}"


def main() -> None:
    by_term = load_all()
    fit = [p for t in FIT_TERMS for p in by_term.get(t, [])]
    test = by_term.get(TEST_TERM, [])

    base_k, base_n = score(rule_always_petitioner, test)
    base = 100 * base_k / base_n
    print(f"OT2025-26 test set: {len({p['docket'] for p in test})} dockets, {len(test)} runs")
    print(f"Majority-class baseline on test: {base:.1f}%")

    # ---------------------------------------------------------- fixed rules
    print("\n" + "=" * 76)
    print("FIXED RULES (no free parameters) — replicate-level accuracy")
    print("=" * 76)
    fixed = [
        ("current (deliberated majority)", rule_current),
        ("no deliberation (initial vote)", rule_no_deliberation),
        ("persona majority (recomputed)", rule_persona_majority),
        ("confidence-weighted personas", rule_conf_weighted),
        ("swing four only", make_subset_rule(SWING)),
        ("always petitioner", rule_always_petitioner),
    ]
    for label, r in fixed:
        k, n = score(r, test)
        print(line(label, k, n, base))

    print("\n  -- same rules, docket-level consensus of the 4 runs --")
    for label, r in fixed:
        k, n = consensus_score(r, test)
        print(line(label, k, n, 100 * base_k / base_n))

    # ---------------------------------------------------------- by model
    print("\n" + "=" * 76)
    print("BY MODEL, AND MODEL ENSEMBLE (OT2025-26)")
    print("=" * 76)
    for m in sorted({p["llm_model"] for p in test}):
        sub = [p for p in test if p["llm_model"] == m]
        k, n = score(rule_current, sub)
        print(line(m, k, n, base))
    # agreement-gated: only count dockets where both models' modal calls agree
    by_docket = defaultdict(lambda: defaultdict(list))
    for p in test:
        by_docket[p["docket"]][p["llm_model"]].append(p["predicted_winner"])
    agree_k = agree_n = dis_k = dis_n = 0
    truth_by_docket = {p["docket"]: p["_truth"] for p in test}
    for d, mm in by_docket.items():
        modal = {m: modal_winner(v) for m, v in mm.items()}
        vals = set(modal.values())
        if len(vals) == 1:
            agree_k += int(vals.pop() == truth_by_docket[d]); agree_n += 1
        else:
            dis_n += 1
    print(line("dockets where models agree", agree_k, agree_n, base))
    print(f"  dockets where models disagree      (n={dis_n:4d})  — no joint call")

    # ---------------------------------------------------------- tuned rules
    print("\n" + "=" * 76)
    print("TUNED RULES — parameter chosen on OT2022-24, applied to OT2025-26")
    print("=" * 76)
    best_k, best_score = None, -1
    for k_ in range(1, 10):
        r = make_threshold_rule(k_)
        kk, nn = score(r, fit)
        acc = 100 * kk / nn
        if acc > best_score:
            best_score, best_k = acc, k_
        print(f"  fit  OT22-24  respondent needs >={k_} of 9: {acc:5.1f}%")
    print(f"\n  -> selected k={best_k} on the fit terms ({best_score:.1f}%)")
    r = make_threshold_rule(best_k)
    kk, nn = score(r, test)
    print(line(f"  TEST OT2025-26 with k={best_k}", kk, nn, base))

    # confidence gate, tuned on fit terms
    print("\n  confidence gate: use the model only above a cutoff, else petitioner")
    best_c, best_c_score = None, -1
    for c in [0.0, 0.70, 0.75, 0.80, 0.85, 0.90]:
        def gate(p, c=c):
            ac = p.get("average_confidence")
            if isinstance(ac, (int, float)) and ac < c:
                return "Petitioner"
            return p.get("predicted_winner")
        kk, nn = score(gate, fit)
        acc = 100 * kk / nn
        if acc > best_c_score:
            best_c_score, best_c = acc, c
        print(f"    fit  OT22-24  cutoff {c:.2f}: {acc:5.1f}%")
    print(f"\n  -> selected cutoff={best_c:.2f} on fit terms ({best_c_score:.1f}%)")

    def gate(p, c=best_c):
        ac = p.get("average_confidence")
        if isinstance(ac, (int, float)) and ac < c:
            return "Petitioner"
        return p.get("predicted_winner")
    kk, nn = score(gate, test)
    print(line(f"  TEST OT2025-26 cutoff={best_c:.2f}", kk, nn, base))

    # ---------------------------------------------------------- oracle
    print("\n" + "=" * 76)
    print("HINDSIGHT CEILING — tuned directly on OT2025-26 (NOT a forecast)")
    print("=" * 76)
    best = []
    for k_ in range(1, 10):
        kk, nn = score(make_threshold_rule(k_), test)
        best.append((100 * kk / nn, f"respondent needs >={k_} of 9"))
    best.sort(reverse=True)
    for acc, lab in best[:3]:
        print(f"  {lab:<34s} {acc:5.1f}%   {acc-base:+5.1f} vs baseline")
    print("\n  Read this block as the most hindsight could buy, not as a result.")

    # ---------------------------------------------------------- selective
    print("\n" + "=" * 76)
    print("SELECTIVE PREDICTION — does the pipeline know when it knows? (OT2025-26)")
    print("=" * 76)
    unan = [p for p in test if len({v.get("vote") for v in (p.get("justice_votes") or {}).values()}) == 1]
    split_ = [p for p in test if p not in unan]
    for lab, sub in [("all 9 personas agree", unan), ("personas split", split_)]:
        if sub:
            kk, nn = score(rule_current, sub)
            cov = 100 * len(sub) / len(test)
            print(line(f"{lab} (coverage {cov:.0f}%)", kk, nn, base))


if __name__ == "__main__":
    main()
