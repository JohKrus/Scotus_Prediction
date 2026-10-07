"""Task 0: reproduce the paper's Table 5 (case accuracy), Table 8 (deliberation
effect) and Table 10 (OT2025 'Actual' column and consensus counts) from the saved
predictions and SCDB 2026_01. No API calls.

    python -m revision.t0_reproduce          (from code/)
"""
from __future__ import annotations

import re

import pandas as pd

from revision import common as C

PAPER_TABLE5 = {  # (term, model): replicate-level %, paper Table 5
    (2022, "Claude-4.6"): 81.9, (2022, "GPT-5.2"): 88.8, (2022, "Gemini-2.5"): 71.5,
    (2023, "Claude-4.6"): 86.4, (2023, "GPT-5.2"): 80.5, (2023, "Gemini-2.5"): 72.7,
    (2024, "Claude-4.6"): 65.6, (2024, "GPT-5.2"): 66.4,
    (2025, "Claude-4.6"): 57.0, (2025, "GPT-5.2"): 63.2,
}
PAPER_BASELINE = {2022: 62.1, 2023: 72.9, 2024: 70.3, 2025: 66.7}
PAPER_TABLE8 = {  # first round, final, winner changed %, flips/case
    (2022, "Claude-4.6"): (81.9, 81.9, 0.0, 0.29), (2022, "GPT-5.2"): (88.8, 88.8, 0.0, 0.04),
    (2022, "Gemini-2.5"): (72.4, 71.6, 2.6, 0.32), (2023, "Claude-4.6"): (85.6, 86.4, 0.8, 0.24),
    (2023, "GPT-5.2"): (80.5, 80.5, 0.0, 0.04), (2023, "Gemini-2.5"): (71.2, 72.7, 4.5, 0.32),
    (2024, "Claude-4.6"): (66.4, 65.6, 0.8, 0.27), (2024, "GPT-5.2"): (66.4, 66.4, 0.0, 0.05),
    (2025, "Claude-4.6"): (58.8, 57.0, 1.8, 0.41), (2025, "GPT-5.2"): (63.2, 63.2, 0.0, 0.07),
}
TABLE10_TEX = C.CODE_DIR / "revision" / "paper_refs" / "ot2526_predictions_table_v3.tex"


def net_changes(row) -> int:
    """Justices whose final vote differs from their initial vote."""
    fin, ini = row["justice_votes"], row["initial_votes"]
    return sum(1 for n in fin if n in ini and fin[n]["vote"] != ini[n]["vote"])


def any_round_changes(row) -> int:
    """Justices who changed at least once across rounds."""
    rounds = row["votes_per_round"]
    changed = set()
    for a, b in zip(rounds, rounds[1:]):
        changed |= {n for n in b if n in a and a[n]["vote"] != b[n]["vote"]}
    return len(changed)


def parse_table10(path) -> pd.DataFrame:
    rows = []
    for line in open(path, encoding="utf-8"):
        m = re.match(r"\s*(\S+?)(\\add\{\$\^\{\\mathrm\{(\w)\}\}\$\})?\s*&(.*)&(.*)&(.*)\\\\", line)
        if not m or m.group(1).startswith("\\"):
            continue
        docket, note = m.group(1), m.group(3)

        def side(cell):
            s = re.search(r"(Pet|Resp)\. \((\d+-\d+)\)", cell)
            return (("Petitioner" if s.group(1) == "Pet" else "Respondent"), s.group(2)) if s else (None, None)

        cw, cs = side(m.group(4)); gw, gs = side(m.group(5)); aw, a_s = side(m.group(6))
        rows.append(dict(docket=docket, note=note, claude_cons=cw, claude_split=cs,
                         gpt_cons=gw, gpt_split=gs, actual=aw, actual_split=a_s,
                         claude_mark="checkmark" in m.group(4), gpt_mark="checkmark" in m.group(5)))
    return pd.DataFrame(rows).set_index("docket")


def main():
    o = C.out("t0_reproduce")
    runs = C.scored_runs()
    truth = C.truth_table()

    # ---- Table 5 ---------------------------------------------------------
    t5 = []
    for (term, model), g in runs.groupby(["term", "model"]):
        p, lo, hi = C.boot_mean(g, "correct")
        cons = g.groupby("docket").predicted_winner.apply(C.modal_winner)
        cons_acc = (cons == truth.loc[cons.index, "winner"]).mean()
        t5.append(dict(term=term, model=model, n_cases=g.docket.nunique(), n_runs=len(g),
                       acc=C.pct(p), lo=C.pct(lo), hi=C.pct(hi), consensus_acc=C.pct(cons_acc),
                       paper=PAPER_TABLE5.get((term, model))))
    t5 = pd.DataFrame(t5)
    t5["diff_vs_paper"] = (t5.acc - t5.paper).round(1)
    pred_cases = truth.loc[runs.docket.unique()]   # scored cases that were forecast
    base = (pred_cases.groupby("term").winner.apply(lambda s: (s == "Petitioner").mean() * 100)
            .round(1).rename("baseline").reset_index())
    base["n_cases"] = pred_cases.groupby("term").size().values
    base["paper"] = base.term.map(PAPER_BASELINE)
    t5.to_csv(o / "table5_reproduction.csv", index=False)
    base.to_csv(o / "baseline_reproduction.csv", index=False)

    # ---- Table 8 ---------------------------------------------------------
    runs["net_changes"] = runs.apply(net_changes, axis=1)
    runs["any_round_changes"] = runs.apply(any_round_changes, axis=1)
    t8 = []
    for (term, model), g in runs.groupby(["term", "model"]):
        pr = PAPER_TABLE8.get((term, model), (None,) * 4)
        t8.append(dict(term=term, model=model,
                       first_round=C.pct(g.initial_correct.mean()), final=C.pct(g.correct.mean()),
                       winner_changed=C.pct((g.initial_winner != g.predicted_winner).mean()),
                       flips_total_vote_changes=round(g.total_vote_changes.mean(), 2),
                       flips_net=round(g.net_changes.mean(), 3),
                       flips_any_round=round(g.any_round_changes.mean(), 2),
                       paper_first=pr[0], paper_final=pr[1], paper_changed=pr[2], paper_flips=pr[3]))
    t8 = pd.DataFrame(t8)
    t8.to_csv(o / "table8_reproduction.csv", index=False)

    # ---- Table 10 --------------------------------------------------------
    lines = []
    if TABLE10_TEX.exists():
        t10 = parse_table10(TABLE10_TEX)
        sc = C.scdb_cases()
        t10["scdb_winner"] = [sc.winner.get(d) for d in t10.index]
        t10["scdb_split"] = [sc.split.get(d) for d in t10.index]
        scored = t10[t10.actual.notna()]
        mism = scored[(scored.actual != scored.scdb_winner) | (scored.actual_split != scored.scdb_split)]
        t10.to_csv(o / "table10_vs_scdb.csv")
        lines.append(f"Table 10 scored rows: {len(scored)}; winner+split mismatches vs SCDB 2026_01: {len(mism)}")
        if len(mism):
            lines.append(mism[["actual", "actual_split", "scdb_winner", "scdb_split"]].to_string())
        # consensus counts, ties to first replicate (Table 10 note)
        r25 = runs[runs.term == 2025].sort_values("replicate")
        for model in ("Claude-4.6", "GPT-5.2"):
            g = r25[r25.model == model]
            first = g.groupby("docket").predicted_winner.agg(
                lambda s: s.mode().iloc[0] if s.nunique() == 1 else s.iloc[0])
            k = int((first == truth.loc[first.index, "winner"]).sum())
            tiepet = g.groupby("docket").predicted_winner.apply(C.modal_winner)
            k2 = int((tiepet == truth.loc[tiepet.index, "winner"]).sum())
            ties = int((g.groupby("docket").predicted_winner.nunique() > 1).sum())
            lines.append(f"{model}: consensus correct (ties->first rep) {k}/{len(first)}; "
                         f"(ties->Petitioner) {k2}/{len(tiepet)}; dockets with split replicates {ties}")
    else:
        lines.append(f"Table 10 source not found at {TABLE10_TEX}")

    # ---- OT2022-24 under the release the paper used (2025_01) -------------
    old = C.REPO_DIR / "analysis" / "data" / "SCDB_2025_01_caseCentered_Citation.csv"
    if old.exists():
        o25 = pd.read_csv(old, encoding="latin-1", low_memory=False)
        o25 = o25[o25.term >= 2022].drop_duplicates("docket").set_index("docket")
        o25["winner"] = o25.partyWinning.map({1.0: "Petitioner", 0.0: "Respondent"})
        n26 = C.scdb_cases("2026_01")
        both = o25.index.intersection(n26[n26.term <= 2024].index)
        diff = [d for d in both if o25.winner.get(d) != n26.winner.get(d)]
        lines.append(f"OT2022-24 winner codes differing 2025_01 vs 2026_01: {len(diff)} {diff}")

    with open(o / "reproduction_notes.txt", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    pd.set_option("display.width", 200)
    print(t5.to_string(index=False)); print(base.to_string(index=False))
    print(t8.to_string(index=False)); print("\n".join(lines))


if __name__ == "__main__":
    main()
