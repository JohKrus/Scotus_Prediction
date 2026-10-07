"""Task 6: memorization fingerprints, crowd gap, Kitagawa decomposition, pooled
logit and prominence dose-response (no API).

Amicus counts come from Task 1's inventory (retained files whose filing
description or filename marks an amicus brief), with a first-page check for
dockets without filing metadata (most of OT2025).

    python -m revision.t6_fingerprints
"""
from __future__ import annotations

import re

import fitz
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from revision import common as cm   # not `C`: patsy formulas use C()

FANTASY = {2022: 75.0, 2023: 83.1, 2024: 76.4, 2025: 94.7}   # Term-level crowd accuracy, not case-matched
ISSUE_GROUP = {
    "Criminal Procedure": "Criminal Procedure",
    "Civil Rights": "Rights", "First Amendment": "Rights", "Due Process": "Rights", "Privacy": "Rights",
    "Judicial Power": "Judicial Power",
}   # everything else -> "Economic & other"
_AMICUS_RE = re.compile(r"amic(?:us|i)\s+curiae", re.I)


def issue_group(name) -> str:
    return ISSUE_GROUP.get(name, "Economic & other")


def amicus_counts() -> pd.Series:
    inv = pd.read_csv(cm.OUT_DIR / "t1_input_audit" / "input_inventory.csv", low_memory=False)
    inv = inv[inv.retained]
    has_meta = inv.groupby("docket").description.apply(lambda s: s.notna().any())
    flag = inv.is_amicus.copy()
    # dockets without filing metadata: read the cover page
    for i, r in inv[~inv.docket.map(has_meta)].iterrows():
        f = cm.CODE_DIR / "data" / r.term_dir / f"{r.docket}_pdfs" / r.file
        try:
            with fitz.open(f) as d:
                head = d[0].get_text()[:3000] if len(d) else ""
            flag.loc[i] = bool(_AMICUS_RE.search(head)) or flag.loc[i]
        except Exception:
            pass
    inv["is_amicus2"] = flag
    return inv.groupby("docket").is_amicus2.sum()


def dissenters(votes: dict, winner: str) -> set:
    return {n for n, v in votes.items() if v != winner}


def main():
    o = cm.out("t6_fingerprints")
    runs = cm.scored_runs()
    sc = cm.scdb_cases()
    jv = cm.scdb_justice_votes()
    actual = jv.pivot_table(index="docket", columns="justice_full", values="side", aggfunc="first")
    am = amicus_counts()
    am.rename("amicus_count").to_csv(o / "amicus_counts.csv")

    runs["actual_split"] = runs.docket.map(sc.split)
    runs["actual_min"] = runs.docket.map(sc.minVotes).astype(int)
    runs["pred_min"] = runs.vote_split.str.split("-").str[1].astype(int)
    runs["split_exact"] = (runs.vote_split == runs.actual_split).astype(float)
    runs["minority_size_exact"] = (runs.pred_min == runs.actual_min).astype(float)

    jac, xb_rows = [], []
    for i, r in runs.iterrows():
        pv = {n: v["vote"] for n, v in r.justice_votes.items()}
        av = actual.loc[r.docket].dropna().to_dict() if r.docket in actual.index else {}
        if r.actual_min > 0 and av:
            pdis = dissenters({n: pv[n] for n in av if n in pv}, r.predicted_winner)
            adis = dissenters(av, r.truth)
            jac.append((i, len(pdis & adis) / len(pdis | adis) if pdis | adis else 1.0))
            # bloc-consistency of each actual vote
            for bloc in (cm.CONSERVATIVE, cm.LIBERAL):
                sides = [av[n] for n in bloc if n in av]
                pet = sides.count("Petitioner"); res = len(sides) - pet
                if pet == res:
                    continue
                bmaj = "Petitioner" if pet > res else "Respondent"
                for n in bloc:
                    if n in av and n in pv:
                        xb_rows.append(dict(docket=r.docket, term=r.term, model=r.model, replicate=r.replicate,
                                            justice=n, cross_bloc=av[n] != bmaj, correct=float(pv[n] == av[n])))
    runs["dissenter_jaccard"] = pd.Series(dict(jac))
    xb = pd.DataFrame(xb_rows)

    rows = []
    for (term, model), g in runs.groupby(["term", "model"]):
        r = dict(term=term, model=model, cases=g.docket.nunique(), runs=len(g))
        r["case_acc"], r["case_acc_lo"], r["case_acc_hi"] = cm.boot_mean(g, "correct")
        r["split_exact"], r["split_exact_lo"], r["split_exact_hi"] = cm.boot_mean(g, "split_exact")
        gc = g[g.correct == 1]
        r["split_exact_given_correct"], r["split_exact_gc_lo"], r["split_exact_gc_hi"] = cm.boot_mean(gc, "split_exact")
        r["minority_size_exact"] = g.minority_size_exact.mean()
        gn = g[g.dissenter_jaccard.notna()]
        r["dissenter_jaccard"], r["dissenter_jaccard_lo"], r["dissenter_jaccard_hi"] = cm.boot_mean(gn, "dissenter_jaccard")
        r["dissenter_jaccard_given_correct"] = gn[gn.correct == 1].dissenter_jaccard.mean()
        x = xb[(xb.term == term) & (xb.model == model)]
        xc = x[x.cross_bloc]
        r["cross_bloc_votes"] = len(xc)
        if len(xc):
            r["cross_bloc_acc"], r["cross_bloc_acc_lo"], r["cross_bloc_acc_hi"] = cm.boot_mean(xc, "correct")
        r["bloc_consistent_acc"] = x[~x.cross_bloc].correct.mean()
        piv = g.pivot_table(index="docket", columns="replicate", values=["predicted_winner", "vote_split"], aggfunc="first")
        if piv.shape[1] >= 4:
            r["replicate_agree_winner"] = (piv["predicted_winner"][1] == piv["predicted_winner"][2]).mean()
            r["replicate_agree_split"] = (piv["vote_split"][1] == piv["vote_split"][2]).mean()
        r["mean_confidence"] = g.average_confidence.mean()
        base = (g.drop_duplicates("docket").truth == "Petitioner").mean()
        r["baseline"] = base
        r["gap_vs_baseline_pts"] = 100 * (r["case_acc"] - base)
        r["fantasyscotus"] = FANTASY[term] / 100
        r["gap_vs_crowd_pts_not_case_matched"] = 100 * r["case_acc"] - FANTASY[term]
        rows.append(r)
    fp = pd.DataFrame(rows)
    fp[fp.select_dtypes("number").columns] = fp.select_dtypes("number").round(3)
    fp.to_csv(o / "fingerprints_by_term.csv", index=False)
    print(fp.T.to_string())

    # ------------------------------------------------------------------ Kitagawa
    runs["issue_group"] = runs.docket.map(sc.issue_name).map(issue_group)
    mp = sc.loc[runs.docket.unique()].assign(group=lambda d: d.issue_name.map(issue_group))
    pd.crosstab(mp.group, mp.term).to_csv(o / "issue_group_counts.csv")
    pd.Series(ISSUE_GROUP).rename("group").to_csv(o / "issue_group_mapping.csv")
    kit = kitagawa(runs)
    kit.to_csv(o / "kitagawa.csv", index=False)
    print(kit.to_string(index=False))

    # ------------------------------------------------------------------ pooled logit
    runs["amicus"] = runs.docket.map(am).fillna(0)
    runs["log_amicus"] = np.log1p(runs.amicus)
    lcd = runs.docket.map(sc.lcDispositionDirection)
    runs["lc_direction"] = lcd.map({1.0: "conservative", 2.0: "liberal", 3.0: "unspecifiable"}).fillna("missing")
    runs["state_source"] = (runs.docket.map(sc.caseSource) >= 300).astype(int)
    runs["ot2025"] = (runs.term == 2025).astype(int)
    runs.drop(columns=["justice_votes", "initial_votes", "votes_per_round", "case_analysis"]).to_csv(
        o / "run_level_covariates.csv", index=False)
    res = []
    runs["term_model"] = runs.term.astype(str) + "_" + runs.model   # Term x model cells (Gemini has 2)
    f = ("correct ~ C(term_model, Treatment('2022_GPT-5.2')) + C(issue_group, Treatment('Economic & other')) "
         "+ C(lc_direction, Treatment('conservative')) + state_source + log_amicus")
    rr = runs[runs.model.isin(["GPT-5.2", "Claude-4.6", "Gemini-2.5"])]
    fit = smf.logit(f, rr).fit(disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(rr.docket)[0]})
    with open(o / "pooled_logit.txt", "w") as fh:
        fh.write(fit.summary().as_text())
    tab = pd.DataFrame(dict(coef=fit.params, se=fit.bse, p=fit.pvalues)).round(4)
    tab.to_csv(o / "pooled_logit_coefs.csv")
    print(tab.to_string())

    # ------------------------------------------------------------------ prominence dose-response
    for model, g in runs[runs.model.isin(["GPT-5.2", "Claude-4.6"])].groupby("model"):
        # the OT2025 main effect is absorbed by the Term fixed effects
        fit = smf.logit("correct ~ log_amicus + log_amicus:ot2025 + C(term)", g).fit(
            disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(g.docket)[0]})
        for k in ("log_amicus", "log_amicus:ot2025"):
            res.append(dict(model=model, term=k, coef=fit.params[k], se=fit.bse[k], p=fit.pvalues[k]))
        # implied slope on OT2025 = log_amicus + interaction
        v = fit.cov_params()
        s25 = fit.params["log_amicus"] + fit.params["log_amicus:ot2025"]
        se25 = np.sqrt(v.loc["log_amicus", "log_amicus"] + v.loc["log_amicus:ot2025", "log_amicus:ot2025"]
                       + 2 * v.loc["log_amicus", "log_amicus:ot2025"])
        res.append(dict(model=model, term="slope_OT2025", coef=s25, se=se25, p=np.nan))
    dr = pd.DataFrame(res).round(4)
    dr.to_csv(o / "prominence_logit.csv", index=False)
    print(dr.to_string(index=False))
    prominence_plot(runs, o)
    # does prominence predict division? (difficulty channel)
    cases = sc.loc[runs.docket.unique()].assign(amicus=lambda d: d.index.map(am).fillna(0))
    cases["divided"] = cases.minVotes > 0
    cases["amicus_bin"] = pd.qcut(np.log1p(cases.amicus), 4, labels=False, duplicates="drop")
    cases.groupby(["amicus_bin"]).agg(n=("divided", "size"), divided=("divided", "mean"),
                                      amicus_median=("amicus", "median")).round(3).to_csv(o / "amicus_vs_division.csv")


def kitagawa(runs: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(cm.SEED)

    def parts(dfa, dfb):
        a = dfa.groupby("issue_group").correct.mean(); b = dfb.groupby("issue_group").correct.mean()
        wa = dfa.groupby("issue_group").docket.nunique(); wa = wa / wa.sum()
        wb = dfb.groupby("issue_group").docket.nunique(); wb = wb / wb.sum()
        gs = sorted(set(a.index) | set(b.index))
        a, b, wa, wb = (x.reindex(gs).fillna(0) for x in (a, b, wa, wb))
        comp = ((wb - wa) * (a + b) / 2).sum(); within = ((b - a) * (wa + wb) / 2).sum()
        # total uses docket-weighted means so that comp + within = total exactly
        tot = (wb * b).sum() - (wa * a).sum()
        return tot, comp, within

    out = []
    for model in ("GPT-5.2", "Claude-4.6"):
        g = runs[runs.model == model]
        base = g[g.term.isin([2022, 2023])]
        for tgt in (2024, 2025):
            tg = g[g.term == tgt]
            tot, comp, within = parts(base, tg)
            boots = []
            ba = [x for _, x in base.groupby("docket")]; ta = [x for _, x in tg.groupby("docket")]
            for _ in range(cm.N_BOOT):
                bb = pd.concat([ba[i] for i in rng.integers(0, len(ba), len(ba))])
                tb = pd.concat([ta[i] for i in rng.integers(0, len(ta), len(ta))])
                boots.append(parts(bb, tb))
            boots = np.array(boots)
            lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
            out.append(dict(model=model, comparison=f"OT2022-23 -> OT{tgt}",
                            total_change=round(100 * tot, 1), total_lo=round(100 * lo[0], 1), total_hi=round(100 * hi[0], 1),
                            composition=round(100 * comp, 1), composition_lo=round(100 * lo[1], 1), composition_hi=round(100 * hi[1], 1),
                            within_group=round(100 * within, 1), within_lo=round(100 * lo[2], 1), within_hi=round(100 * hi[2], 1)))
    return pd.DataFrame(out)


def prominence_plot(runs, o):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = runs[runs.model.isin(["GPT-5.2", "Claude-4.6"])].copy()
    d["period"] = np.where(d.term == 2025, "OT2025", "OT2022-24")
    edges = np.quantile(d.drop_duplicates("docket").log_amicus, [0, .25, .5, .75, 1])
    d["bin"] = np.clip(np.searchsorted(edges, d.log_amicus, side="right") - 1, 0, 3)
    tab = d.groupby(["model", "period", "bin"]).agg(acc=("correct", "mean"), n=("correct", "size"),
                                                   amicus_median=("amicus", "median")).reset_index()
    tab.to_csv(o / "prominence_binned.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (model, g) in zip(axes, tab.groupby("model")):
        for period, gg in g.groupby("period"):
            ax.plot(gg.bin, gg.acc, marker="o", label=period)
            for _, r in gg.iterrows():
                ax.annotate(f"n={r.n}", (r.bin, r.acc), fontsize=6, textcoords="offset points", xytext=(0, 5))
        ax.set_xticks(range(4)); ax.set_xticklabels(["Q1 (fewest)", "Q2", "Q3", "Q4 (most)"])
        ax.set_title(model); ax.set_xlabel("amicus-brief count quartile"); ax.legend()
    axes[0].set_ylabel("case accuracy (replicate level)")
    fig.tight_layout(); fig.savefig(o / "prominence_dose_response.png", dpi=150)


if __name__ == "__main__":
    main()
