"""Task 7: recall x prediction merge, within-Term mediation, and sensitivity of
the OT2025 headline (no API; needs Task 1, 3 and 4 outputs).

    python -m revision.t7_mediation
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from revision import common as cm   # not `C`: patsy formulas use C()
from revision.t6_fingerprints import issue_group

RECALL_MODEL = {"GPT-5.2": "GPT-5.2", "Claude-4.6": "Claude-4.6", "Claude-Opus-5.5": "Claude-Opus-5.5"}


def load_single_pass(tag="std") -> pd.DataFrame:
    f = cm.OUT_DIR / "t4_single_pass" / f"predictions_{tag}_scored.csv"
    return pd.read_csv(f, dtype={"docket": str}) if f.exists() else pd.DataFrame()


def main():
    o = cm.out("t7_mediation")
    sc = cm.scdb_cases()
    st = pd.read_csv(cm.OUT_DIR / "t3_recall" / "recall_status_by_docket.csv", dtype={"docket": str})
    st["recall_correct"] = (st.status == "recalled_correct").astype(int)
    pipe = cm.scored_runs()[["docket", "term", "model", "replicate", "correct", "initial_correct", "truth"]]
    pipe = pipe.assign(system="pipeline")
    sp = load_single_pass()
    if len(sp):
        sp = sp.rename(columns={"rep": "replicate"})
        sp["system"] = "single_pass_" + sp.arm
        sp = sp[["docket", "term", "model", "replicate", "correct", "truth", "system"]]
    allp = pd.concat([pipe, sp], ignore_index=True)
    allp["issue_group"] = allp.docket.map(sc.issue_name).map(issue_group)
    allp = allp.merge(st[["model", "docket", "status", "recall_correct", "recall_correct_share"]],
                      on=["model", "docket"], how="left")
    allp.to_csv(o / "merged_runs.csv", index=False)

    # 1. accuracy by recall status within Term
    by = (allp.dropna(subset=["status"]).groupby(["system", "model", "term", "status"])
          .agg(cases=("docket", "nunique"), runs=("correct", "size"), acc=("correct", "mean")).reset_index())
    by["acc"] = by.acc.round(3)
    by.to_csv(o / "accuracy_by_recall_status.csv", index=False)

    # 2. within-Term logit, clustered by docket
    rows = []
    for (system, model), g in allp.dropna(subset=["recall_correct"]).groupby(["system", "model"]):
        if g.term.nunique() < 2 or g.recall_correct.nunique() < 2:
            continue
        try:
            fit = smf.logit("correct ~ recall_correct + C(term) + C(issue_group)", g).fit(disp=0, cov_type="cluster",
                                                            cov_kwds={"groups": pd.factorize(g.docket)[0]})
            ame = fit.get_margeff(at="overall").summary_frame().loc["recall_correct"]
            rows.append(dict(system=system, model=model, n_runs=len(g), n_cases=g.docket.nunique(),
                             coef_recall_correct=round(fit.params["recall_correct"], 3),
                             se=round(fit.bse["recall_correct"], 3), p=round(fit.pvalues["recall_correct"], 4),
                             ame_pts=round(100 * ame["dy/dx"], 1), ame_lo=round(100 * ame["Conf. Int. Low"], 1),
                             ame_hi=round(100 * ame["Cont. Int. Hi."], 1)))
        except Exception as e:
            rows.append(dict(system=system, model=model, error=str(e)[:120]))
    pd.DataFrame(rows).to_csv(o / "mediation_logit.csv", index=False)

    # 3. sensitivity of the OT2025 headline
    pc = pd.read_csv(cm.OUT_DIR / "t1_input_audit" / "prereg_check.csv", dtype={"docket": str})
    decided_before_commit = set(pc[pc.scored & ~pc.committed_before_decision].docket)
    a25 = allp[allp.term == 2025].copy()
    a25["decision_date"] = a25.docket.map(sc.dateDecision)
    variants = {
        "all scored": lambda g: g,
        "excl. decided before 2026-02-01": lambda g: g[g.decision_date >= "2026-02-01"],
        "excl. decided before forecast commit (prereg_check)": lambda g: g[~g.docket.isin(decided_before_commit)],
        "excl. recalled correctly (same model, Task 3)": lambda g: g[g.recall_correct != 1],
        "excl. all three": lambda g: g[(g.decision_date >= "2026-02-01") & ~g.docket.isin(decided_before_commit)
                                       & (g.recall_correct != 1)],
    }
    rows = []
    for (system, model), g in a25.groupby(["system", "model"]):
        for name, f in variants.items():
            h = f(g)
            if not len(h):
                continue
            p, lo, hi = cm.boot_mean(h, "correct")
            base = (h.drop_duplicates("docket").truth == "Petitioner").mean()
            d = h.groupby("docket").correct.mean() - (h.groupby("docket").truth.first() == "Petitioner")
            dp, dlo, dhi = cm.fast_bootstrap_mean([np.array([v]) for v in d])
            rows.append(dict(system=system, model=model, variant=name, cases=h.docket.nunique(),
                             acc=cm.pct(p), lo=cm.pct(lo), hi=cm.pct(hi), baseline=cm.pct(base),
                             diff_vs_baseline=cm.pct(dp), diff_lo=cm.pct(dlo), diff_hi=cm.pct(dhi),
                             applies_to_model="yes" if "2026-02-01" not in name or model.startswith("Claude-4.6") else "reference only"))
    sens = pd.DataFrame(rows)
    sens.to_csv(o / "ot2025_sensitivity.csv", index=False)
    pd.set_option("display.width", 250)
    print(by.to_string(index=False)); print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
