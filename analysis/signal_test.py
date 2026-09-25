"""Is there any usable signal in the v2 forecasts, and how should OT2026-27 be judged?

Two tests the accuracy tables cannot answer:

1. AUC. Accuracy conflates the decision threshold with the underlying signal. If
   the persona petitioner-count ranks cases better than chance, a recalibrated
   threshold could help even when raw accuracy is poor. AUC ~ 0.5 means there is
   nothing to recalibrate.

2. Forward-chaining evaluation. The four terms are ordered in time and the model's
   edge over baseline trends with recency, so leave-one-term-out CV (which lets
   future terms inform past ones) overstates what to expect next. Fitting only on
   terms strictly before the test term is the right analogue for OT2026-27.

Run from repo root:  python v2/code/signal_test.py
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from optimize_rules import (ALL_TERMS, load_all, make_threshold,  # noqa: E402
                            persona_votes, rule_always_pet, rule_current, score)
from score_predictions import wilson  # noqa: E402


def auc(scores: list[float], labels: list[int]) -> float:
    """Rank-based AUC (ties averaged). labels: 1 = petitioner won."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for t in range(i, j + 1):
            ranks[order[t]] = avg
        i = j + 1
    rsum = sum(r for r, y in zip(ranks, labels) if y == 1)
    return (rsum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def hdr(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)


def main() -> None:
    by_term = load_all()

    hdr("A. AUC — does the persona vote count rank cases at all?")
    print("  Score = number of personas voting Petitioner (0-9), pooled per docket.")
    print("  0.50 = no ranking information.\n")
    for t in ALL_TERMS:
        recs = by_term.get(t, [])
        if not recs:
            continue
        agg = defaultdict(list)
        for p in recs:
            v = persona_votes(p)
            if v:
                agg[p["docket"]].append(sum(x == "Petitioner" for x in v.values()))
        s, y = [], []
        truth = {p["docket"]: p["_truth"] for p in recs}
        for d, vals in agg.items():
            s.append(sum(vals) / len(vals))
            y.append(int(truth[d] == "Petitioner"))
        a = auc(s, y)
        print(f"  OT{t}: AUC {a:.3f}   (n={len(s)} dockets, {sum(y)} petitioner wins)")

    print("\n  Same, using stated average_confidence as the score:")
    for t in ALL_TERMS:
        recs = by_term.get(t, [])
        agg, truth = defaultdict(list), {}
        for p in recs:
            c = p.get("average_confidence")
            if isinstance(c, (int, float)):
                signed = c if p["predicted_winner"] == "Petitioner" else -c
                agg[p["docket"]].append(signed)
                truth[p["docket"]] = p["_truth"]
        if not agg:
            continue
        s = [sum(v) / len(v) for v in agg.values()]
        y = [int(truth[d] == "Petitioner") for d in agg]
        print(f"  OT{t}: AUC {auc(s, y):.3f}")

    hdr("B. EDGE OVER BASELINE BY TERM — the contamination signature")
    print(f"  {'term':<8}{'model':>9}{'baseline':>11}{'edge':>9}")
    edges = []
    for t in ALL_TERMS:
        recs = by_term.get(t, [])
        if not recs:
            continue
        ck, cn = score(rule_current, recs)
        bk, bn = score(rule_always_pet, recs)
        c, b = 100 * ck / cn, 100 * bk / bn
        edges.append((t, c - b))
        print(f"  OT{t:<6}{c:8.1f}%{b:10.1f}%{c-b:+9.1f}")
    print("\n  edge trend: " + " -> ".join(f"{e:+.1f}" for _, e in edges))
    slope = (edges[-1][1] - edges[0][1]) / (len(edges) - 1)
    print(f"  mean change per term: {slope:+.1f} points")

    hdr("C. FORWARD-CHAINING — fit only on earlier terms, test on the next")
    print("  This is the OT2026-27 analogue. k is chosen on all strictly-earlier terms.\n")
    tk = tn = bk_ = bn_ = 0
    for i, t in enumerate(ALL_TERMS):
        if i == 0:
            continue
        fit = [p for tt in ALL_TERMS[:i] for p in by_term.get(tt, [])]
        test = by_term.get(t, [])
        if not test or not fit:
            continue
        best_k, best = None, -1.0
        for k_ in range(1, 10):
            kk, nn = score(make_threshold(k_), fit)
            if nn and 100 * kk / nn > best:
                best, best_k = 100 * kk / nn, k_
        kk, nn = score(make_threshold(best_k), test)
        bb, bn2 = score(rule_always_pet, test)
        tk += kk; tn += nn; bk_ += bb; bn_ += bn2
        print(f"  train <OT{t} -> pick k={best_k} | test OT{t}: {100*kk/nn:5.1f}%  "
              f"vs baseline {100*bb/bn2:5.1f}%  ({100*kk/nn - 100*bb/bn2:+.1f})")
    p, lo, hi = wilson(tk, tn)
    pb, _, _ = wilson(bk_, bn_)
    print(f"\n  Pooled forward-chained: {p:.1f}%  [{lo:.1f}, {hi:.1f}]  vs baseline {pb:.1f}%"
          f"  ({p-pb:+.1f})")
    print("  Compare with the leave-one-term-out figure (+4.4): letting later terms")
    print("  inform earlier folds inflates the estimate.")

    hdr("D. WHAT THE MOST RECENT TERM IMPLIES FOR OT2026-27")
    recs25 = by_term.get(2025, [])
    ck, cn = score(rule_current, recs25)
    bk2, bn2 = score(rule_always_pet, recs25)
    print(f"  OT2025-26 is the only uncontaminated term: model {100*ck/cn:.1f}%, "
          f"baseline {100*bk2/bn2:.1f}%")
    print("  Best available point estimate for an unchanged pipeline on OT2026-27:")
    print(f"    roughly {100*ck/cn:.0f}%, i.e. about {100*bk2/bn2 - 100*ck/cn:.0f} points BELOW")
    print("    the majority-class baseline. Any claimed improvement has to clear that bar.")


if __name__ == "__main__":
    main()
