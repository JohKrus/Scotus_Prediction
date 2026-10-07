"""Task 5: deliberation mechanics and institutional benchmarks (no API).

Reads the saved pipeline runs (predictions/term_*/, read-only) and SCDB 2026_01.
Groups: model x {OT2022-24 pooled, OT2025}; by-Term tables are written too.

    python -m revision.t5_deliberation
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from revision import common as C

LOCK = 0.85


def group_of(term: int) -> str:
    return "OT2022-24" if term <= 2024 else "OT2025"


def run_level(r: pd.Series) -> dict:
    rounds = r.votes_per_round
    init = rounds[0]
    fin = r.justice_votes
    names = [n for n in C.JUSTICES if n in fin and n in init]
    init_pet = sum(init[n]["vote"] == "Petitioner" for n in names)
    init_major = "Petitioner" if init_pet > len(names) - init_pet else "Respondent"
    out = dict(n_rounds=len(rounds) - 1)
    # flips per round (accepted changes between consecutive rounds)
    for k in range(1, 4):
        out[f"flips_r{k}"] = (sum(rounds[k][n]["vote"] != rounds[k - 1][n]["vote"] for n in rounds[k])
                              if k < len(rounds) else np.nan)
    # locks: lock test runs at the start of round >= 2 on the previous round's confidence
    r1 = rounds[1] if len(rounds) > 1 else {}
    out["share_conf_ge_085_initial"] = np.mean([init[n]["confidence"] >= LOCK for n in names])
    out["share_locked_after_r1"] = np.mean([r1[n]["confidence"] >= LOCK for n in r1]) if r1 else np.nan
    out["share_final_flag_r1"] = np.mean([bool(r1[n]["is_final"]) for n in r1]) if r1 else np.nan
    last = rounds[-1]
    out["share_final_flag_last"] = np.mean([bool(last[n]["is_final"]) for n in last])
    out["share_final_below_lock_last"] = np.mean([bool(last[n]["is_final"]) and last[n]["confidence"] < LOCK
                                                  for n in last])
    # termination reason, reconstructed from should_continue_deliberating()
    if out["n_rounds"] >= 3:
        r2 = rounds[2]
        pet2 = sum(v["vote"] == "Petitioner" for v in r2.values())
        margin2 = abs(2 * pet2 - len(r2))
        ch2 = out["flips_r2"]
        out["termination"] = "max_rounds(close_margin)" if margin2 <= 3 else (
            "max_rounds(change_in_r2)" if ch2 else "max_rounds(other)")
    else:
        out["termination"] = "converged_after_r2(margin>=5,no_change)"
    out["any_round_with_2_flips"] = any(out[f"flips_r{k}"] == 2 for k in (1, 2, 3)
                                        if not np.isnan(out[f"flips_r{k}"]))
    # switching
    sw = [n for n in names if fin[n]["vote"] != init[n]["vote"]]
    any_flip = set()
    for a, b in zip(rounds, rounds[1:]):
        any_flip |= {n for n in b if n in a and a[n]["vote"] != b[n]["vote"]}
    out["switch_rate_net"] = len(sw) / len(names)
    out["switch_rate_any"] = len(any_flip) / len(names)
    out["n_switches"] = len(sw)
    out["n_switches_toward_initial_majority"] = sum(fin[n]["vote"] == init_major for n in sw)
    out["winner_changed"] = r.initial_winner != r.predicted_winner
    pet = sum(fin[n]["vote"] == "Petitioner" for n in names)
    out["pred_minority"] = min(pet, len(names) - pet)
    out["roberts_in_majority"] = (fin.get("John Roberts", {}).get("vote") == r.predicted_winner)
    return out


def split_cat(minority: int) -> str:
    return {0: "unanimous", 1: "1 dissent", 2: "2 dissents", 3: "3 dissents", 4: "4 dissents"}.get(int(minority), "?")


def main():
    o = C.out("t5_deliberation")
    runs = C.scored_runs()
    sc = C.scdb_cases()
    jv = C.scdb_justice_votes()
    lvl = pd.DataFrame([run_level(r) for _, r in runs.iterrows()], index=runs.index)
    d = pd.concat([runs[["docket", "term", "model", "replicate", "predicted_winner", "truth"]], lvl], axis=1)
    d["group"] = d.term.map(group_of)
    d["actual_minority"] = d.docket.map(sc.minVotes).astype(int)
    d["actual_unanimous"] = d.actual_minority == 0
    d["pred_unanimous"] = d.pred_minority == 0
    d.to_csv(o / "run_level.csv", index=False)

    def summarize(g):
        sw = g.n_switches.sum()
        tp = (g.pred_unanimous & g.actual_unanimous).sum()
        out = dict(
            runs=len(g), cases=g.docket.nunique(),
            share_conf_ge_085_initial=g.share_conf_ge_085_initial.mean(),
            share_locked_after_r1=g.share_locked_after_r1.mean(),
            share_final_flag_r1=g.share_final_flag_r1.mean(),
            share_final_flag_last=g.share_final_flag_last.mean(),
            share_final_below_lock_last=g.share_final_below_lock_last.mean(),
            share_2_rounds=(g.n_rounds == 2).mean(), share_3_rounds=(g.n_rounds == 3).mean(),
            flips_r1=g.flips_r1.mean(), flips_r2=g.flips_r2.mean(), flips_r3=g.flips_r3.mean(),
            share_runs_with_a_2_flip_round=g.any_round_with_2_flips.mean(),
            switch_rate_net=g.switch_rate_net.mean(), switch_rate_any=g.switch_rate_any.mean(),
            share_switches_toward_initial_majority=(g.n_switches_toward_initial_majority.sum() / sw) if sw else np.nan,
            n_switches=int(sw), winner_changed=g.winner_changed.mean(),
            pred_unanimity=g.pred_unanimous.mean(), actual_unanimity=g.actual_unanimous.mean(),
            precision_pred9_0=tp / g.pred_unanimous.sum() if g.pred_unanimous.sum() else np.nan,
            recall_pred9_0=tp / g.actual_unanimous.sum() if g.actual_unanimous.sum() else np.nan,
            roberts_in_pred_majority=g.roberts_in_majority.mean(),
        )
        for k in range(5):
            out[f"pred_{split_cat(k).replace(' ', '_')}"] = (g.pred_minority == k).mean()
            out[f"actual_{split_cat(k).replace(' ', '_')}"] = (g.actual_minority == k).mean()
        for t, n in g.termination.value_counts(normalize=True).items():
            out[f"term_{t}"] = n
        return pd.Series(out)

    tab = d.groupby(["group", "model"]).apply(summarize).reset_index()
    tab_term = d.groupby(["term", "model"]).apply(summarize).reset_index()
    # actual Roberts-in-majority rate on the same cases
    rob = jv[jv.justice_full == "John Roberts"].set_index("docket").majority.eq(2.0)
    for t in (tab, tab_term):
        key = "group" if "group" in t else "term"
        t["roberts_in_actual_majority"] = [
            rob.reindex(d[(d[key] == r[key]) & (d.model == r.model)].docket).mean() for _, r in t.iterrows()]
    num = tab.select_dtypes("number").columns
    tab[num] = tab[num].round(3)
    tab_term[tab_term.select_dtypes("number").columns] = tab_term.select_dtypes("number").round(3)
    tab.to_csv(o / "deliberation_by_group.csv", index=False)
    tab_term.to_csv(o / "deliberation_by_term.csv", index=False)

    # bootstrap CIs for the headline switching and unanimity measures
    ci = []
    for (grp, model), g in d.groupby(["group", "model"]):
        for col in ("switch_rate_net", "winner_changed", "pred_unanimous", "actual_unanimous"):
            p, lo, hi = C.boot_mean(g.assign(v=g[col].astype(float)), "v")
            ci.append(dict(group=grp, model=model, metric=col, est=round(p, 4), lo=round(lo, 4), hi=round(hi, 4)))
    pd.DataFrame(ci).to_csv(o / "deliberation_ci.csv", index=False)

    agreement(runs, jv, o)
    print(tab.T.to_string())


def agreement(runs, jv, o):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    short = [n.split()[-1] for n in C.JUSTICES]
    actual = jv.pivot_table(index="docket", columns="justice_full", values="side", aggfunc="first")
    rows, mats = [], {}
    runs = runs.assign(group=runs.term.map(group_of))
    for (grp, model), g in runs.groupby(["group", "model"]):
        # predicted: per replicate, then average over replicates
        pm = np.zeros((9, 9)); reps = sorted(g.replicate.unique())
        for rep in reps:
            gr = g[g.replicate == rep]
            V = pd.DataFrame([{n: v["vote"] for n, v in r.justice_votes.items()} for _, r in gr.iterrows()])
            for i, a in enumerate(C.JUSTICES):
                for j, b in enumerate(C.JUSTICES):
                    ok = V[a].notna() & V[b].notna()
                    pm[i, j] += (V.loc[ok, a] == V.loc[ok, b]).mean() / len(reps)
        A = actual.reindex(g.docket.unique())
        am = np.zeros((9, 9))
        for i, a in enumerate(C.JUSTICES):
            for j, b in enumerate(C.JUSTICES):
                ok = A[a].notna() & A[b].notna()
                am[i, j] = (A.loc[ok, a] == A.loc[ok, b]).mean()
        iu = np.triu_indices(9, 1)
        rows.append(dict(group=grp, model=model, cases=g.docket.nunique(),
                         corr_offdiag=round(np.corrcoef(pm[iu], am[iu])[0, 1], 3),
                         mad_offdiag=round(np.abs(pm[iu] - am[iu]).mean(), 3),
                         mean_pred_agreement=round(pm[iu].mean(), 3), mean_actual_agreement=round(am[iu].mean(), 3),
                         pred_cross_bloc=round(np.mean([pm[i, j] for i, a in enumerate(C.JUSTICES) for j, b in enumerate(C.JUSTICES)
                                                        if i < j and ((a in C.LIBERAL) != (b in C.LIBERAL))]), 3),
                         actual_cross_bloc=round(np.mean([am[i, j] for i, a in enumerate(C.JUSTICES) for j, b in enumerate(C.JUSTICES)
                                                          if i < j and ((a in C.LIBERAL) != (b in C.LIBERAL))]), 3)))
        mats[(grp, model)] = (pm, am)
        pd.DataFrame(pm, index=short, columns=short).round(3).to_csv(o / f"agreement_pred_{grp}_{model}.csv")
        pd.DataFrame(am, index=short, columns=short).round(3).to_csv(o / f"agreement_actual_{grp}_{model}.csv")
    pd.DataFrame(rows).to_csv(o / "agreement_summary.csv", index=False)
    keys = sorted(mats)
    fig, axes = plt.subplots(len(keys), 2, figsize=(9, 4.1 * len(keys)), squeeze=False)
    for row, k in zip(axes, keys):
        for ax, M, lab in zip(row, mats[k], ("predicted (final votes)", "actual (SCDB)")):
            im = ax.imshow(M, vmin=0.3, vmax=1, cmap="viridis")
            ax.set_xticks(range(9)); ax.set_yticks(range(9))
            ax.set_xticklabels(short, rotation=60, fontsize=7); ax.set_yticklabels(short, fontsize=7)
            ax.set_title(f"{k[1]} {k[0]}: {lab}", fontsize=9)
            for i in range(9):
                for j in range(9):
                    ax.text(j, i, f"{M[i, j]:.2f}"[1:] if M[i, j] < 1 else "1", ha="center", va="center",
                            fontsize=5.5, color="w" if M[i, j] < 0.7 else "k")
    fig.colorbar(im, ax=axes, shrink=0.3, label="share of cases voting the same side")
    fig.savefig(o / "agreement_heatmaps.png", dpi=140, bbox_inches="tight")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
