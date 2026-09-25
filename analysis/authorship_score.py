"""Score opinion-authorship and separate-writing forecasts against the Court's record.

Truth: SCDB (`majOpinWriter`, `vote`, `opinion`) for OT2019-2024; the corrected
OT2025-26 results sheet (parse_writings) for OT2025. Forecasts: the cached output
of authorship_predict.py. Baselines need no case reading:

  Majority author (argued cases with a signed opinion of the Court)
    uniform         - each member of the ACTUAL majority equally likely.
    sitting balance - the Court-watcher heuristic: among the actual majority, the
                      justices who have written the fewest opinions of the Court from
                      the same argument sitting, counting only cases decided before
                      this one. It uses information released after argument, so it is
                      a reference point, not an ex-ante rival.
  Separate writings (per justice-case)
    prior rate      - the justice's rate of writing separately / concurring over the
                      three prior Terms.

The model's author forecast is scored unconditionally (probability 0 if the real
author is outside its predicted majority) and on runs that got the winner right.

Run from repo root:  python v2/code/authorship_score.py            (baselines work before any forecasts exist)
"""
from __future__ import annotations

import glob
import json
import os
import random
import sys
from collections import defaultdict

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from paths import DATA_DIR, PRED_DIR  # noqa: E402

from dissent_prediction import auc  # noqa: E402
from parse_justice_votes_ot2526 import JUSTICES, PERSONA_TO_SURNAME, load_writings_truth  # noqa: E402
from score_predictions import CASE_CSV, JUSTICE_CSV, PRED_GLOB  # noqa: E402

FORECASTS = os.path.join(PRED_DIR, "authorship", "term_*", "*.json")
TERMS = (2022, 2023, 2024, 2025)
SCDB_ID = {108: "Thomas", 111: "Roberts", 112: "Alito", 113: "Sotomayor", 114: "Kagan",
           115: "Gorsuch", 116: "Kavanaugh", 117: "Barrett", 118: "Jackson"}
SITTING_GAP_DAYS = 14   # argument days more than two weeks apart start a new sitting
VOTE_ROLE = {1: "join", 5: "join", 8: "join", 2: "dissent", 6: "dissent", 7: "dissent",
             3: "concur", 4: "concur_judgment"}


# ----------------------------------------------------------------------------- truth
def load_cases() -> dict[str, dict]:
    """docket -> {term, argued, decided, author, roles, wrote} for OT2019-2025."""
    cc = pd.read_csv(CASE_CSV, encoding="latin-1", low_memory=False,
                     usecols=["term", "docket", "dateArgument", "dateDecision", "partyWinning"])
    cc = cc[cc.term >= 2019].drop_duplicates("docket")
    jc = pd.read_csv(JUSTICE_CSV, encoding="latin-1", low_memory=False,
                     usecols=["term", "docket", "justice", "vote", "opinion", "majOpinWriter"])
    jc = jc[jc.term >= 2019].drop_duplicates(["docket", "justice"])

    cases = {}
    for r in cc.itertuples(index=False):
        cases[str(r.docket)] = {"term": int(r.term), "argued": pd.to_datetime(r.dateArgument),
                                "decided": pd.to_datetime(r.dateDecision),
                                "winner": {1.0: "Petitioner", 0.0: "Respondent"}.get(r.partyWinning),
                                "author": None, "roles": {}, "wrote": {}}
    for r in jc.itertuples(index=False):
        c = cases.get(str(r.docket))
        sur = SCDB_ID.get(r.justice)
        if c is None or sur is None or pd.isna(r.vote):
            continue
        if pd.notna(r.majOpinWriter):
            c["author"] = SCDB_ID.get(int(r.majOpinWriter))
        c["roles"][sur] = VOTE_ROLE.get(int(r.vote), "join")
        c["wrote"][sur] = r.opinion in (2.0, 3.0)
    for c in cases.values():
        if c["author"]:
            c["roles"][c["author"]] = "author"
            c["wrote"][c["author"]] = False   # the opinion of the Court is not a separate writing

    x = pd.read_excel(os.path.join(DATA_DIR, "scotus_ot2025_cases.xlsx"))
    dates = {str(d).strip(): (a, dd, win) for d, a, dd, win in
             zip(x["Docket number"], x["Date argued"], x["Date decided"],
                 x["Winner (petitioner/respondent)"])}
    for d, w in load_writings_truth().items():
        a, dd, win = dates[d]
        cases[d] = {"term": 2025, "argued": pd.to_datetime(a), "decided": pd.to_datetime(dd),
                    "winner": win,
                    "author": w["author"], "wrote": w["wrote"],
                    "roles": {j: r for j, r in w["roles"].items() if r != "recused"}}
        if w["author"]:
            cases[d]["wrote"][w["author"]] = False
    return cases


def assign_sittings(cases) -> None:
    """Add a 'sitting' index within each Term by clustering argument dates."""
    for t in {c["term"] for c in cases.values()}:
        days = sorted({c["argued"] for c in cases.values() if c["term"] == t and pd.notna(c["argued"])})
        sitting, s = {}, 0
        for i, day in enumerate(days):
            if i and (day - days[i - 1]).days > SITTING_GAP_DAYS:
                s += 1
            sitting[day] = s
        for c in cases.values():
            if c["term"] == t:
                c["sitting"] = sitting.get(c["argued"]) if pd.notna(c["argued"]) else None


def majority_of(c) -> list[str]:
    return [j for j, r in c["roles"].items() if r != "dissent"]


def sitting_balance_hit(d, c, cases) -> float:
    """Expected top-1 accuracy of the sitting-balance heuristic for case d."""
    written = defaultdict(int)
    for o in cases.values():
        if (o["term"] == c["term"] and o.get("sitting") == c["sitting"] and o["author"]
                and pd.notna(o["decided"]) and o["decided"] < c["decided"]):
            written[o["author"]] += 1
    maj = majority_of(c)
    fewest = min(written[j] for j in maj)
    cand = [j for j in maj if written[j] == fewest]
    return (c["author"] in cand) / len(cand)


def prior_rates(cases) -> dict[int, dict[str, tuple[float, float]]]:
    """Term -> justice -> (rate of writing separately, rate of concurring), prior three Terms."""
    out = {}
    for t in TERMS:
        acc = defaultdict(lambda: [0, 0, 0])
        for c in cases.values():
            if t - 3 <= c["term"] <= t - 1:
                for j, r in c["roles"].items():
                    acc[j][0] += c["wrote"].get(j, False)
                    acc[j][1] += r in ("concur", "concur_judgment")
                    acc[j][2] += 1
        n_all = sum(v[2] for v in acc.values())
        pooled = (sum(v[0] for v in acc.values()) / n_all, sum(v[1] for v in acc.values()) / n_all)
        out[t] = {j: (acc[j][0] / acc[j][2], acc[j][1] / acc[j][2]) if acc[j][2] else pooled
                  for j in JUSTICES}
    return out


# ----------------------------------------------------------------------------- scoring
def predicted_dockets() -> set[str]:
    out = set()
    for f in glob.glob(PRED_GLOB):
        out.add(os.path.basename(f).split("_")[0])
    return out


def load_forecasts():
    fc = []
    for f in glob.glob(FORECASTS):
        r = json.load(open(f, encoding="utf-8"))
        r["author_p"] = {PERSONA_TO_SURNAME[k]: v for k, v in r["majority_author"].items()}
        r["sep"] = {PERSONA_TO_SURNAME[k]: v for k, v in r["separate_writings"].items()}
        fc.append(r)
    return fc


def boot_ci(by_case: dict[str, list[float]], reps=2000, seed=7):
    rng = random.Random(seed)
    keys = sorted(by_case)
    if not keys:
        return float("nan"), (float("nan"), float("nan"))

    def mean(ks):
        v = [x for k in ks for x in by_case[k]]
        return sum(v) / len(v)
    draws = sorted(mean([rng.choice(keys) for _ in keys]) for _ in range(reps))
    return mean(keys), (draws[int(0.025 * reps)], draws[int(0.975 * reps) - 1])


def main():
    cases = load_cases()
    assign_sittings(cases)
    rates = prior_rates(cases)
    ours = predicted_dockets()

    print("Sittings per Term (argued cases per sitting):")
    for t in TERMS:
        n = defaultdict(int)
        for c in cases.values():
            if c["term"] == t and c.get("sitting") is not None:
                n[c["sitting"]] += 1
        print(f"  OT{t}: " + " ".join(str(n[s]) for s in sorted(n)))

    print("\nBASELINES on our forecast cases (argued, signed opinion of the Court)")
    print(f"  {'term':6s} {'cases':>5s} {'uniform':>8s} {'sitting':>8s}")
    for t in TERMS:
        sel = [d for d, c in cases.items() if c["term"] == t and d in ours and c["author"]
               and c.get("sitting") is not None]
        uni = sum(1 / len(majority_of(cases[d])) for d in sel) / len(sel)
        sb = sum(sitting_balance_hit(d, cases[d], cases) for d in sel) / len(sel)
        print(f"  OT{t} {len(sel):5d} {100*uni:7.1f}% {100*sb:7.1f}%")

    fc = [r for r in load_forecasts() if r["docket"] in cases]
    if not fc:
        print("\nNo authorship forecasts cached yet -- run authorship_predict.py first.")
        return

    print("\nMAJORITY AUTHOR: model vs. baselines (per run; Claude + GPT pooled rows add a case-bootstrap CI)")
    print(f"  {'term':6s} {'model':11s} {'runs':>5s} {'top-1':>6s} {'P(auth)':>8s} "
          f"{'top-1|W':>8s} {'uniform':>8s} {'sitting':>8s}")
    groups = defaultdict(list)
    for r in fc:
        c = cases[r["docket"]]
        if not c["author"] or c.get("sitting") is None:
            continue
        top = max(r["author_p"], key=r["author_p"].get)
        groups[(c["term"], r["llm_model"])].append({
            "d": r["docket"], "top1": float(top == c["author"]),
            "pa": r["author_p"].get(c["author"], 0.0),
            "win_ok": r["predicted_winner"] == c["winner"],
            "uni": 1 / len(majority_of(c)), "sb": sitting_balance_hit(r["docket"], c, cases)})
    for t in TERMS:
        for m in ("Claude-4.6", "GPT-5.2"):
            g = groups.get((t, m))
            if not g:
                continue
            w = [x for x in g if x["win_ok"]]
            f = lambda k, xs: 100 * sum(x[k] for x in xs) / len(xs) if xs else float("nan")
            print(f"  OT{t} {m:11s} {len(g):5d} {f('top1', g):5.1f}% {f('pa', g):7.1f}% "
                  f"{f('top1', w):7.1f}% {f('uni', g):7.1f}% {f('sb', g):7.1f}%")
        pooled = groups.get((t, "Claude-4.6"), []) + groups.get((t, "GPT-5.2"), [])
        if pooled:
            by = defaultdict(list)
            for x in pooled:
                by[x["d"]].append(x["top1"] - x["uni"])
            est, ci = boot_ci(by)
            print(f"  OT{t} pooled: top-1 minus uniform {100*est:+.1f} pts [{100*ci[0]:+.1f}, {100*ci[1]:+.1f}]")

    print("\nSEPARATE WRITINGS (per justice-case): AUC of model probability vs. prior-Term rate")
    print(f"  {'term':6s} {'model':11s} {'n':>5s} {'write%':>6s} {'AUCw':>6s} {'AUCw-prior':>10s} "
          f"{'conc%':>6s} {'AUCc':>6s} {'AUCc-prior':>10s}")
    for t in TERMS:
        for m in ("Claude-4.6", "GPT-5.2"):
            rows = []
            for r in fc:
                c = cases[r["docket"]]
                if c["term"] != t or r["llm_model"] != m or r.get("coerced_entries"):
                    continue   # coerced records lack a real "concurs" forecast
                for j, role in c["roles"].items():
                    if role == "author" or j not in r["sep"]:
                        continue
                    rows.append((r["sep"][j]["writes_separately"], rates[t][j][0], c["wrote"].get(j, False),
                                 r["sep"][j]["concurs"], rates[t][j][1], role in ("concur", "concur_judgment")))
            if not rows:
                continue
            pw = lambda i, lab: [x[i] for x in rows if x[lab]]
            nw = lambda i, lab: [x[i] for x in rows if not x[lab]]
            print(f"  OT{t} {m:11s} {len(rows):5d} {100*sum(x[2] for x in rows)/len(rows):5.1f}% "
                  f"{auc(pw(0, 2), nw(0, 2)):6.3f} {auc(pw(1, 2), nw(1, 2)):10.3f} "
                  f"{100*sum(x[5] for x in rows)/len(rows):5.1f}% "
                  f"{auc(pw(3, 5), nw(3, 5)):6.3f} {auc(pw(4, 5), nw(4, 5)):10.3f}")


if __name__ == "__main__":
    main()
