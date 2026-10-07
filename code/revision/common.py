"""Shared loaders for the revision diagnostics.

Everything here is read-only with respect to the pre-registration record:
predictions are read from ../predictions/ (repo root) and never written.

Ground truth comes from the Supreme Court Database, release 2026_01, docket-
organized files (case- and justice-centered), which cover OT2022-OT2025. Put the
four CSVs in code/data/scdb/ (third-party data, not committed):
    SCDB_2026_01_caseCentered_Docket.csv, SCDB_2026_01_justiceCentered_Docket.csv
"""
from __future__ import annotations

import glob
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parent.parent          # code/
REPO_DIR = CODE_DIR.parent                                  # repo root
PRED_DIR = REPO_DIR / "predictions"                         # read-only
SCDB_DIR = Path(os.environ.get("SCDB_DIR", CODE_DIR / "data" / "scdb"))
OUT_DIR = CODE_DIR / "results" / "revision"

SEED = 20261007
N_BOOT = 2000

TERM_FOLDERS = {"term_2022_2023": 2022, "term_2023_2024": 2023,
                "term_2024_2025": 2024, "term_2025_2026": 2025}

# prediction justice name -> SCDB justiceName
JUSTICE_CODE = {
    "John Roberts": "JGRoberts", "Clarence Thomas": "CThomas", "Samuel Alito": "SAAlito",
    "Sonia Sotomayor": "SSotomayor", "Elena Kagan": "EKagan", "Neil Gorsuch": "NMGorsuch",
    "Brett Kavanaugh": "BMKavanaugh", "Amy Coney Barrett": "ACBarrett",
    "Ketanji Brown Jackson": "KBJackson",
}
JUSTICES = list(JUSTICE_CODE)
CONSERVATIVE = {"John Roberts", "Clarence Thomas", "Samuel Alito", "Neil Gorsuch",
                "Brett Kavanaugh", "Amy Coney Barrett"}
LIBERAL = {"Sonia Sotomayor", "Elena Kagan", "Ketanji Brown Jackson"}

# OT2025 dockets excluded from scoring, as in the paper (Table 10 notes a, b)
OT2025_EXCLUDE = {
    "24-872": "dismissed as improvidently granted",
    "25-170": "carried over to OT2026", "25-459": "carried over to OT2026",
    "25-498": "carried over to OT2026", "25-579": "carried over to OT2026",
}

# Published training-data cutoffs (provider documentation), used only for plot markers
PUBLISHED_CUTOFF = {
    "GPT-5.2": "2025-08-31",
    "Claude-4.6": "2025-08-31",          # reliable knowledge; training data through Jan 2026
    "Claude-4.6 (training data)": "2026-01-31",
    "Gemini-2.5": "2025-01-31",
}

ISSUE_AREA = {
    1: "Criminal Procedure", 2: "Civil Rights", 3: "First Amendment", 4: "Due Process",
    5: "Privacy", 6: "Attorneys", 7: "Unions", 8: "Economic Activity", 9: "Judicial Power",
    10: "Federalism", 11: "Interstate Relations", 12: "Federal Taxation",
    13: "Miscellaneous", 14: "Private Action",
}


# ---------------------------------------------------------------------------
# SCDB
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def scdb_cases(release: str = "2026_01") -> pd.DataFrame:
    """Case-centered, docket-organized SCDB, one row per docket, OT2022 onward."""
    f = SCDB_DIR / f"SCDB_{release}_caseCentered_Docket.csv"
    if not f.exists():
        raise FileNotFoundError(f"SCDB file missing: {f}")
    c = pd.read_csv(f, encoding="latin-1", low_memory=False)
    c = c[c.term >= 2022].copy()
    c["docket"] = c["docket"].astype(str).str.strip()
    for col in ("dateDecision", "dateArgument"):
        c[col] = pd.to_datetime(c[col], format="%m/%d/%Y", errors="coerce")
    c["winner"] = c["partyWinning"].map({1.0: "Petitioner", 0.0: "Respondent"})
    c["split"] = c["majVotes"].astype("Int64").astype(str) + "-" + c["minVotes"].astype("Int64").astype(str)
    c["unanimous"] = c["minVotes"] == 0
    c["close"] = (c["majVotes"] - c["minVotes"]) <= 1
    c["issue_name"] = c["issueArea"].map(ISSUE_AREA)
    return c.drop_duplicates("docket").set_index("docket", drop=False)


@lru_cache(maxsize=None)
def scdb_justice_votes(release: str = "2026_01") -> pd.DataFrame:
    """(docket, justice) -> side voted for: in-majority XOR petitioner-won.

    Same proxy as analysis/score_predictions.py. Justices not participating
    (majority NaN) are dropped.
    """
    f = SCDB_DIR / f"SCDB_{release}_justiceCentered_Docket.csv"
    j = pd.read_csv(f, encoding="latin-1", low_memory=False)
    j = j[j.term >= 2022].copy()
    j["docket"] = j["docket"].astype(str).str.strip()
    cases = scdb_cases(release)
    j = j[j.docket.isin(cases.index)]
    j["winner"] = j.docket.map(cases["winner"])
    j = j[j.majority.isin([1.0, 2.0]) & j.winner.notna()]
    in_maj = j.majority == 2.0
    pet_won = j.winner == "Petitioner"
    j["side"] = np.where(in_maj == pet_won, "Petitioner", "Respondent")
    code_to_name = {v: k for k, v in JUSTICE_CODE.items()}
    j["justice_full"] = j.justiceName.map(code_to_name)
    j = j[j.justice_full.notna()]
    return j[["docket", "term", "justice_full", "side", "majority", "opinion"]].drop_duplicates(
        ["docket", "justice_full"])


def truth_table(release: str = "2026_01") -> pd.DataFrame:
    """Scored cases: SCDB winner known and (for OT2025) not excluded."""
    c = scdb_cases(release)
    c = c[c.winner.notna()].copy()
    c = c[~((c.term == 2025) & c.docket.isin(OT2025_EXCLUDE))]
    return c


# ---------------------------------------------------------------------------
# Predictions (read-only)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def load_runs() -> pd.DataFrame:
    """One row per saved run (docket x model x replicate) from predictions/term_*/."""
    rows = []
    for folder, fterm in TERM_FOLDERS.items():
        for f in sorted(glob.glob(str(PRED_DIR / folder / "*_predictions.json"))):
            with open(f, encoding="utf-8") as fh:
                arr = json.load(fh)
            for i, d in enumerate(arr):
                rows.append({
                    "folder": folder, "folder_term": fterm, "file": os.path.basename(f),
                    "docket": d.get("docket"), "model": d.get("llm_model"),
                    "replicate": d.get("replicate", i + 1),
                    "predicted_winner": d.get("predicted_winner"),
                    "initial_winner": d.get("initial_winner"),
                    "vote_split": d.get("vote_split"),
                    "initial_vote_split": d.get("initial_vote_split"),
                    "petitioner_votes": d.get("petitioner_votes"),
                    "average_confidence": d.get("average_confidence"),
                    "deliberation_rounds": d.get("deliberation_rounds"),
                    "total_vote_changes": d.get("total_vote_changes"),
                    "has_transcripts": d.get("has_transcripts"),
                    "justice_votes": d.get("justice_votes", {}),
                    "initial_votes": d.get("initial_votes", {}),
                    "votes_per_round": d.get("votes_per_round", []),
                    "case_analysis": d.get("case_analysis", {}),
                })
    return pd.DataFrame(rows)


def scored_runs(release: str = "2026_01") -> pd.DataFrame:
    """Runs merged to SCDB truth, grouped by the SCDB (decision) Term."""
    runs = load_runs()
    t = truth_table(release)
    r = runs[runs.docket.isin(t.index)].copy()
    r["term"] = r.docket.map(t["term"]).astype(int)
    r["truth"] = r.docket.map(t["winner"])
    r["correct"] = (r.predicted_winner == r.truth).astype(int)
    r["initial_correct"] = (r.initial_winner == r.truth).astype(int)
    return r


def modal_winner(preds) -> str:
    """Docket consensus; ties go to Petitioner (analysis/score_predictions.modal_winner)."""
    preds = list(preds)
    pet = sum(p == "Petitioner" for p in preds)
    return "Petitioner" if pet >= len(preds) - pet else "Respondent"


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def case_bootstrap(df: pd.DataFrame, stat, cluster: str = "docket",
                   n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float, float, float]:
    """Point estimate and 95% percentile interval, resampling clusters (cases).

    `stat` maps a DataFrame to a float. Resampled clusters keep all their rows.
    """
    point = stat(df)
    groups = {k: g for k, g in df.groupby(cluster, sort=False)}
    keys = np.array(list(groups))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(keys, size=len(keys), replace=True)
        vals.append(stat(pd.concat([groups[k] for k in pick], ignore_index=True)))
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return point, lo, hi


def fast_bootstrap_mean(values_by_case: list[np.ndarray], n_boot: int = N_BOOT,
                        seed: int = SEED) -> tuple[float, float, float]:
    """Pooled mean of row values, resampling cases. Much faster than case_bootstrap."""
    sums = np.array([v.sum() for v in values_by_case], dtype=float)
    cnts = np.array([len(v) for v in values_by_case], dtype=float)
    point = sums.sum() / cnts.sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), size=(n_boot, len(sums)))
    boots = sums[idx].sum(1) / cnts[idx].sum(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, lo, hi


def boot_mean(df: pd.DataFrame, col: str, cluster: str = "docket", **kw):
    groups = [g[col].to_numpy(dtype=float) for _, g in df.groupby(cluster, sort=False)]
    return fast_bootstrap_mean(groups, **kw)


def pct(x: float) -> float:
    return round(100 * x, 1)


def out(task: str) -> Path:
    p = OUT_DIR / task
    p.mkdir(parents=True, exist_ok=True)
    return p
