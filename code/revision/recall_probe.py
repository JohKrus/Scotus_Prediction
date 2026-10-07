"""Task 3: closed-book recall probe with placebos.

Arms
  real        every scored case OT2022-OT2025 that the pipeline forecast
              (caption = SCDB caseName, plus docket number)
  fabricated  50 invented cases per Term, captions and docket numbers in that
              Term's format, seeded; no caption collides with any SCDB caseName
  (the real OT2025 cases double as the natural placebo for models whose
   training data precede the Term)

No retrieval, persona, tools or web search. Prompt text: PROMPT below (also in
PLAN.md). Two replicates per case and model.

    python -m revision.recall_probe build                 # case list only
    python -m revision.recall_probe run --smoke 3         # 3 cases, prints projected cost
    python -m revision.recall_probe run [--models GPT-5.2,Claude-4.6] [--workers 8]
    python -m revision.recall_probe run --models Claude-Opus-5.5 --terms 2025
    python -m revision.recall_probe score
"""
from __future__ import annotations

import argparse
import random
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from revision import common as C
from revision import llm

PROMPT = """Answer from your own knowledge only. Do not guess unless a field asks you to.
Case: {caption}, No. {docket}, Supreme Court of the United States, October Term {year}.
Return only this JSON:
{{
  "recall_case": "yes" | "no",
  "decided_on_merits": "yes" | "no" | "dont_know",
  "prevailing_party": "petitioner" | "respondent" | "dont_know",
  "vote_split": "<e.g. 6-3>" | "dont_know",
  "majority_author": "<surname>" | "dont_know",
  "p_petitioner_wins": <number between 0 and 1; required, give your best estimate even if guessing>
}}"""

REPLICATES = 2
N_FAKE_PER_TERM = 50
FAKE_SEED = 20261007

_SURNAMES = """Halvorsen Pruett Delacroix Ostrowski Whitcombe Marchetti Bancroft Tremblay Okafor Lindqvist
Castellano Pemberton Abernathy Kowalczyk Thibodeaux Vandermeer Galloway Nakashima Ferraro Hollister
Quintero Ashworth Brannigan Dunleavy Esterhazy Fairbanks Grisham Hartigan Ingersoll Jessup Kincaid
Lockhart Mahoney Northcutt Oyelaran Prescott Rademacher Sorensen Tolliver Underhill Valencourt Wexler
Yarborough Zielinski Albrecht Brightwell Calloway Drummond Everhart Fenwick Gallagher Holloway Iverson
Jaramillo Kessler Lindgren McCaskey Nordstrom Ortolano Pelletier Rasmussen Strickland Thornbury
Vasquez-Holt Whitlock Youngblood Szymanski Achterberg Beaumont Corrigan Desrosiers Eckhardt""".split()
_STATES = ["ALABAMA", "ARIZONA", "ARKANSAS", "COLORADO", "CONNECTICUT", "FLORIDA", "GEORGIA", "IDAHO",
           "ILLINOIS", "INDIANA", "IOWA", "KANSAS", "KENTUCKY", "LOUISIANA", "MARYLAND", "MICHIGAN",
           "MINNESOTA", "MISSISSIPPI", "MISSOURI", "MONTANA", "NEBRASKA", "NEVADA", "NEW JERSEY",
           "NEW MEXICO", "OHIO", "OKLAHOMA", "OREGON", "PENNSYLVANIA", "SOUTH CAROLINA", "TENNESSEE",
           "TEXAS", "UTAH", "VIRGINIA", "WASHINGTON", "WISCONSIN", "WYOMING"]
_CO_WORDS = ["MERIDIAN", "BLUESTONE", "CARRAWAY", "NORTHFIELD", "HALCYON", "IRONGATE", "PINECREST",
             "SILVERLINE", "TRISTATE", "KESTREL", "ANVIL", "LAKESHORE", "REDWATER", "SUMMIT RIDGE"]
_CO_KINDS = ["HOLDINGS, INC.", "LOGISTICS, LLC", "ENERGY CORP.", "FINANCIAL SERVICES, INC.",
             "HEALTH SYSTEMS, INC.", "MANUFACTURING CO.", "PHARMACEUTICALS, INC.", "TECHNOLOGIES, INC."]
_AGENCIES = ["SECRETARY OF LABOR", "COMMISSIONER OF INTERNAL REVENUE", "FEDERAL TRADE COMMISSION",
             "DEPARTMENT OF HOMELAND SECURITY", "SECURITIES AND EXCHANGE COMMISSION"]


def _norm(s: str) -> str:
    return re.sub(r"[^A-Z0-9 ]", "", str(s).upper()).strip()


def fabricate(all_scdb: pd.DataFrame) -> pd.DataFrame:
    rng = random.Random(FAKE_SEED)
    names = set(all_scdb.caseName.map(_norm))
    dockets = set(all_scdb.docket.astype(str))
    rows = []
    for term in (2022, 2023, 2024, 2025):
        made = 0
        while made < N_FAKE_PER_TERM:
            kind = rng.random()
            sn = rng.choice(_SURNAMES).upper()
            if kind < 0.45:
                pet, resp = sn, "UNITED STATES" if rng.random() < 0.6 else rng.choice(_STATES)
            elif kind < 0.65:
                pet, resp = sn, rng.choice(_SURNAMES).upper()
            elif kind < 0.85:
                pet = f"{rng.choice(_CO_WORDS)} {rng.choice(_CO_KINDS)}"
                resp = rng.choice([rng.choice(_SURNAMES).upper(), rng.choice(_AGENCIES)])
            else:
                pet, resp = rng.choice(_STATES), rng.choice([sn, f"{rng.choice(_CO_WORDS)} {rng.choice(_CO_KINDS)}"])
            if pet == resp:
                continue
            caption = f"{pet} v. {resp}"
            yy = (term - 2000) - (1 if rng.random() < 0.55 else 0)
            num = rng.randint(100, 1300) if rng.random() < 0.75 else rng.randint(5000, 7999)
            docket = f"{yy}-{num}"
            if _norm(caption) in names or docket in dockets:
                continue
            # neither party pair may co-occur in any real caption
            if any(_norm(pet) in n and _norm(resp) in n for n in names):
                continue
            names.add(_norm(caption)); dockets.add(docket)
            rows.append(dict(arm="fabricated", term=term, docket=docket, caption=caption))
            made += 1
    return pd.DataFrame(rows)


def build_cases() -> pd.DataFrame:
    full = pd.read_csv(C.SCDB_DIR / "SCDB_2026_01_caseCentered_Docket.csv", encoding="latin-1",
                       low_memory=False, usecols=["docket", "caseName", "term"])
    runs = C.scored_runs()
    t = C.truth_table()
    real = t.loc[sorted(runs.docket.unique())]
    real = pd.DataFrame(dict(arm="real", term=real.term.values, docket=real.index,
                             caption=real.caseName.values))
    cases = pd.concat([real, fabricate(full)], ignore_index=True)
    o = C.out("t3_recall")
    cases.to_csv(o / "cases.csv", index=False)
    return cases


def run(models, terms, workers, smoke, arms):
    o = C.out("t3_recall")
    cases = pd.read_csv(o / "cases.csv", dtype={"docket": str}) if (o / "cases.csv").exists() else build_cases()
    cases = cases[cases.term.isin(terms) & cases.arm.isin(arms)]
    if smoke:
        cases = cases.groupby("arm").head(smoke)
    llm.load_keys()
    for model in models:
        cache = llm.Cache(o / "raw" / f"{model}.jsonl")
        jobs = []
        for _, c in cases.iterrows():
            prompt = PROMPT.format(caption=c.caption, docket=c.docket, year=c.term)
            for rep in range(1, REPLICATES + 1):
                key = f"{c.arm}|{c.docket}|{rep}"
                if not cache.get(key):
                    jobs.append((key, c, rep, prompt))
        print(f"{model}: {len(jobs)} calls to make", flush=True)

        def work(j):
            key, c, rep, prompt = j
            r = llm.call(model, prompt)
            cache.put(dict(key=key, arm=c.arm, term=int(c.term), docket=c.docket, rep=rep, model=model,
                           prompt_sha=llm.prompt_hash(prompt), **r))
            return r

        with ThreadPoolExecutor(workers) as ex:
            res = list(ex.map(work, jobs))
        tin = sum(r["in_tok"] for r in res); tout = sum(r["out_tok"] for r in res)
        errs = sum(r["stop"].startswith("error") for r in res)
        spent = llm.cost(model, tin, tout)
        print(f"{model}: {len(res)} calls, {tin} in / {tout} out tokens, ${spent:.2f}, errors {errs}")
        if smoke and res:
            full = cases_all = pd.read_csv(o / "cases.csv")
            n_full = len(cases_all[cases_all.term.isin(terms) & cases_all.arm.isin(arms)]) * REPLICATES
            print(f"  projected full run for {model}: ${spent / len(res) * n_full:.2f} ({n_full} calls)")
            for r in res[:2]:
                print("  sample:", r["text"][:300].replace("\n", " "))


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

_JUSTICE_SURNAME = {"JGRoberts": "roberts", "CThomas": "thomas", "SAAlito": "alito", "SSotomayor": "sotomayor",
                    "EKagan": "kagan", "NMGorsuch": "gorsuch", "BMKavanaugh": "kavanaugh",
                    "ACBarrett": "barrett", "KBJackson": "jackson", "SGBreyer": "breyer"}
_JUSTICE_ID = {108: "thomas", 110: "breyer", 111: "roberts", 112: "alito", 113: "sotomayor",
               114: "kagan", 115: "gorsuch", 116: "kavanaugh", 117: "barrett", 118: "jackson"}  # SCDB justice codes


def _split(s) -> str | None:
    m = re.search(r"(\d)\s*[-–to ]+\s*(\d)", str(s))
    if not m:
        return None
    a, b = sorted(map(int, m.groups()), reverse=True)
    return f"{a}-{b}"


def _dk(v) -> bool:
    return v is None or str(v).strip().lower() in ("dont_know", "don't know", "unknown", "", "none", "null")


def parse_all() -> pd.DataFrame:
    o = C.out("t3_recall")
    rows = []
    for f in sorted((o / "raw").glob("*.jsonl")):
        cache = llm.Cache(f)
        for rec in cache.done.values():
            j = llm.parse_json(rec["text"]) or {}
            p = j.get("p_petitioner_wins")
            try:
                p = float(p)
                p = p / 100 if p > 1 else p
            except (TypeError, ValueError):
                p = np.nan
            rows.append(dict(model=rec["model"], arm=rec["arm"], term=rec["term"], docket=rec["docket"],
                             rep=rec["rep"], parsed=bool(j),
                             recall_yes=str(j.get("recall_case", "")).lower().startswith("y"),
                             merits=str(j.get("decided_on_merits", "")).lower(),
                             party=str(j.get("prevailing_party", "dont_know")).lower(),
                             split=None if _dk(j.get("vote_split")) else _split(j.get("vote_split")),
                             author=None if _dk(j.get("majority_author")) else str(j.get("majority_author")).lower(),
                             p_pet=p))
    return pd.DataFrame(rows)


def score():
    o = C.out("t3_recall")
    d = parse_all()
    sc = C.scdb_cases()
    real = d.arm == "real"
    d.loc[real, "truth"] = d.loc[real, "docket"].map(sc.winner).str.lower()
    d.loc[real, "truth_split"] = d.loc[real, "docket"].map(sc.split)
    d.loc[real, "truth_author"] = d.loc[real, "docket"].map(sc.majOpinWriter).map(_JUSTICE_ID).fillna("per curiam")
    d.loc[real, "decision_date"] = d.loc[real, "docket"].map(sc.dateDecision)
    d["party_answered"] = d.party.isin(["petitioner", "respondent"])
    d["split_answered"] = d.split.notna()
    d["author_answered"] = d.author.notna()
    d["winner_correct"] = (d.party == d.truth)
    d["split_correct"] = (d.split == d.truth_split)
    d["author_correct"] = d.apply(lambda r: isinstance(r.author, str) and isinstance(r.truth_author, str)
                                  and r.truth_author.split()[-1] in r.author, axis=1)
    d["brier"] = (d.p_pet - (d.truth == "petitioner").astype(float)) ** 2
    d.loc[~real, "brier"] = np.nan
    d.to_csv(o / "responses_parsed.csv", index=False)

    rows = []
    for (model, term, arm), g in d.groupby(["model", "term", "arm"]):
        r = dict(model=model, term=term, arm=arm, n_cases=g.docket.nunique(), n_resp=len(g),
                 parse_rate=g.parsed.mean())
        for col in ("recall_yes", "party_answered", "split_answered", "author_answered"):
            r[col], r[col + "_lo"], r[col + "_hi"] = C.boot_mean(g.assign(v=g[col].astype(float)), "v")
        if arm == "real":
            for what in ("winner", "split", "author"):
                ans = g[f"{'party' if what == 'winner' else what}_answered"]
                r[f"{what}_acc_dk_wrong"], r[f"{what}_acc_dk_wrong_lo"], r[f"{what}_acc_dk_wrong_hi"] = \
                    C.boot_mean(g.assign(v=g[f"{what}_correct"].astype(float)), "v")
                ga = g[ans]
                r[f"{what}_acc_dk_missing"] = ga[f"{what}_correct"].mean() if len(ga) else np.nan
            r["brier"], r["brier_lo"], r["brier_hi"] = C.boot_mean(g[g.p_pet.notna()], "brier")
            r["always_pet_brier_0.5"] = 0.25
        rows.append(r)
    s = pd.DataFrame(rows)
    # knowledge = real-arm rate minus fabricated-arm rate (same model and Term)
    for col in ("recall_yes", "party_answered", "split_answered", "author_answered"):
        fab = s[s.arm == "fabricated"].set_index(["model", "term"])[col]
        s.loc[s.arm == "real", f"knowledge_{col}"] = [
            r[col] - fab.get((r.model, r.term), np.nan) for _, r in s[s.arm == "real"].iterrows()]
    num = s.select_dtypes("number").columns.difference(["term", "n_cases", "n_resp"])
    s[num] = s[num].round(3)
    s.to_csv(o / "recall_by_term.csv", index=False)

    # docket-level recall status for Task 7 (modal over replicates)
    rr = d[real].copy()
    rr["status"] = np.where(~rr.party_answered, "dont_know", np.where(rr.winner_correct, "recalled_correct", "recalled_wrong"))
    st = (rr.groupby(["model", "docket"]).agg(
        status=("status", lambda x: x.mode().iloc[0] if x.nunique() == 1 else "mixed"),
        recall_correct_share=("winner_correct", "mean"), claimed_recall_share=("recall_yes", "mean"),
        p_pet=("p_pet", "mean"), term=("term", "first"), decision_date=("decision_date", "first")).reset_index())
    st.to_csv(o / "recall_status_by_docket.csv", index=False)
    cutoff_curve(rr)
    print(s.to_string())


def step_cutoff(g: pd.DataFrame, months) -> tuple:
    """One-break step model: P(correct) = a up to month tau, b after; max likelihood over months."""
    y = g.winner_correct.astype(float).to_numpy()
    mo = g.month.to_numpy()
    best = (-np.inf, None, np.nan, np.nan)
    for tau in months[:-1]:
        left = y[mo <= tau]; right = y[mo > tau]
        if len(left) < 10 or len(right) < 10:
            continue
        ll = 0.0
        for part in (left, right):
            p = np.clip(part.mean(), 1e-6, 1 - 1e-6)
            ll += (part * np.log(p) + (1 - part) * np.log(1 - p)).sum()
        if ll > best[0]:
            best = (ll, tau, left.mean(), right.mean())
    return best[1:]


def estimate_cutoffs(rr: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(C.SEED)
    rows = []
    for model, g in rr.groupby("model"):
        months = np.sort(g.month.unique())
        tau, a, b = step_cutoff(g, months)
        boots = []
        groups = [x for _, x in g.groupby("docket")]
        for _ in range(C.N_BOOT // 4):   # 500 draws: the grid search is slow
            pick = rng.integers(0, len(groups), len(groups))
            bt = step_cutoff(pd.concat([groups[i] for i in pick]), months)[0]
            if bt is not None:
                boots.append(pd.Timestamp(bt).toordinal())
        lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan))
        rows.append(dict(model=model, tau_last_month_before_break=pd.Timestamp(tau).date() if tau is not None else None,
                         acc_before=round(a, 3), acc_after=round(b, 3), reported=bool(tau is not None and b < a),
                         tau_lo=pd.Timestamp.fromordinal(int(lo)).date() if boots else None,
                         tau_hi=pd.Timestamp.fromordinal(int(hi)).date() if boots else None,
                         n_boot=len(boots)))
    return pd.DataFrame(rows)


def cutoff_curve(rr: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def lowess(y, x, frac=0.35):
        """Local-linear tricube smoother (LOWESS without robustness iterations; the
        statsmodels extension is blocked by this machine's application control)."""
        x = np.asarray(x, float); y = np.asarray(y, float)
        k = max(3, int(np.ceil(frac * len(x))))
        fit = np.empty(len(x))
        for i, x0 in enumerate(x):
            d = np.abs(x - x0); h = np.sort(d)[k - 1] or 1.0
            w = np.clip(1 - (d / h) ** 3, 0, None) ** 3
            W = w.sum(); mx = (w * x).sum() / W; my = (w * y).sum() / W
            vx = (w * (x - mx) ** 2).sum()
            b = (w * (x - mx) * (y - my)).sum() / vx if vx > 0 else 0.0
            fit[i] = my + b * (x0 - mx)
        o_ = np.argsort(x)
        return np.column_stack([x[o_], np.clip(fit[o_], 0, 1)])
    o = C.out("t3_recall")
    rr = rr.copy()
    rr["month"] = pd.to_datetime(rr.decision_date).dt.to_period("M").dt.to_timestamp()
    m = (rr.groupby(["model", "month"]).agg(n=("docket", "size"), cases=("docket", "nunique"),
                                            winner_acc=("winner_correct", "mean"),
                                            author_acc=("author_correct", "mean"),
                                            claimed_recall=("recall_yes", "mean"),
                                            answered=("party_answered", "mean")).reset_index())
    m.to_csv(o / "cutoff_curve_monthly.csv", index=False)
    est = estimate_cutoffs(rr)
    est.to_csv(o / "effective_cutoff.csv", index=False)
    print(est.to_string(index=False))
    models = list(m.model.unique())
    fig, axes = plt.subplots(len(models), 1, figsize=(10, 3.4 * len(models)), sharex=True, squeeze=False)
    markers = {"GPT-5.2": [("GPT-5.2 cutoff", "2025-08-31")],
               "Claude-4.6": [("reliable knowledge", "2025-08-31"), ("training data", "2026-01-31")],
               "Claude-Opus-5.5": [("training data", "2026-06-30")]}
    for ax, model in zip(axes[:, 0], models):
        g = rr[rr.model == model].sort_values("decision_date")
        x = pd.to_datetime(g.decision_date).map(pd.Timestamp.toordinal).to_numpy()
        for col, lab, colr in (("winner_correct", "winner correct", "#1f6fb4"),
                               ("author_correct", "author correct", "#c2571a")):
            mm = m[m.model == model]
            ax.scatter(mm.month, mm[col.replace("_correct", "_acc")], s=mm.n * 3, alpha=.35, color=colr)
            if len(g) > 10:
                sm = lowess(g[col].astype(float).to_numpy(), x, frac=0.35)
                ax.plot([pd.Timestamp.fromordinal(int(v)) for v in sm[:, 0]], sm[:, 1], color=colr, label=lab)
        for lab, dt in markers.get(model, []):
            ax.axvline(pd.Timestamp(dt), color="grey", ls="--", lw=1)
            ax.text(pd.Timestamp(dt), 1.02, lab, fontsize=7, ha="center")
        ax.axhline(0.5, color="lightgrey", lw=0.8)
        ax.set_ylim(-0.03, 1.08); ax.set_ylabel(model); ax.legend(loc="lower left", fontsize=8)
    axes[-1, 0].set_xlabel("decision month (SCDB dateDecision); dot size = responses")
    fig.suptitle("Closed-book recall: winner and author accuracy by decision month (dont_know = wrong)")
    fig.tight_layout()
    fig.savefig(o / "cutoff_curve.png", dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "run", "score"])
    ap.add_argument("--models", default="GPT-5.2,Claude-4.6")
    ap.add_argument("--terms", default="2022,2023,2024,2025")
    ap.add_argument("--arms", default="real,fabricated")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "build":
        print(build_cases().groupby(["arm", "term"]).size())
    elif a.cmd == "run":
        run(a.models.split(","), [int(t) for t in a.terms.split(",")], a.workers, a.smoke, a.arms.split(","))
    else:
        score()


if __name__ == "__main__":
    main()
