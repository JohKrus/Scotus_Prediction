"""Forecast opinion authorship and separate writings from the saved deliberation runs.

A post-hoc step on top of the v2 pipeline: each saved run (one model, one
replicate) already holds the case analysis and all nine justices' final votes,
confidence and reasoning. One additional call to the SAME model that produced the
run turns that simulated conference into
  - a probability distribution over the predicted majority for the opinion author;
  - per justice, the probability of writing a separate opinion and of concurring.
The deliberation itself is not re-run, so the vote predictions stay exactly as
pre-registered.

Contamination note: for Terms inside a model's training window this is a recall
probe as much as a forecast -- authorship is hard to infer from briefs and easy
to remember -- which is the point of scoring it by Term (authorship_score.py).

Keys come from environment variables (ANTHROPIC_API_KEY, OPENAI_API_KEY). Output
is cached per run under v2/predictions/authorship/, so interrupted runs resume.

Run from repo root:
  python v2/code/authorship_predict.py --dry-run            # build prompts, no API calls
  python v2/code/authorship_predict.py --limit 4            # small paid pilot
  python v2/code/authorship_predict.py                      # everything not yet cached
Options: --terms 2025 --models Claude-4.6,GPT-5.2 --workers 4
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from paths import PRED_DIR  # noqa: E402

from score_predictions import CASE_CSV, PRED_GLOB, load_scdb  # noqa: E402

OUT_DIR = os.path.join(PRED_DIR, "authorship")
PROMPT_VERSION = "authorship-v1"

# Model label in the saved runs -> (provider, pinned model ID). GPT-5.2 is the
# snapshot the gpt-5.2 alias resolved to for every saved run.
MODELS = {
    "Claude-4.6": ("anthropic", "claude-sonnet-4-6"),
    "GPT-5.2": ("openai", "gpt-5.2-2025-12-11"),
}
TEMPERATURE = 0.3  # as in the deliberation runs
MAX_TOKENS = 4000

# Seniority: the Chief Justice assigns when in the majority, otherwise the most
# senior associate justice in the majority.
SENIORITY = ["John Roberts", "Clarence Thomas", "Samuel Alito", "Sonia Sotomayor",
             "Elena Kagan", "Neil Gorsuch", "Brett Kavanaugh", "Amy Coney Barrett",
             "Ketanji Brown Jackson"]

SYSTEM = (
    "You are an expert forecaster of the U.S. Supreme Court. You are given the record "
    "of a simulated conference in an argued case: a structured analysis of the case and "
    "each justice's vote, confidence and reasoning. Take the simulated votes as given and "
    "forecast how the opinions will be written. Respond with a single JSON object and "
    "nothing else."
)

USER_TEMPLATE = """CASE: {case_name} (No. {docket})

CASE ANALYSIS
Legal question: {focal_point}
Provision at issue: {legal_provision}
Petitioner's position: {petitioner_position}
Respondent's position: {respondent_position}
Key precedents: {key_precedents}
Complexity: {legal_complexity}

SIMULATED CONFERENCE
Predicted disposition: for the {winner}, {split}.
Majority ({n_maj}): {majority}
Minority ({n_min}): {minority}

{votes}

ASSIGNMENT
By convention the Chief Justice assigns the opinion of the Court when in the majority;
otherwise the most senior associate justice in the majority assigns. In this case the
assigning justice would be {assigner}.

TASK
Return JSON with exactly these keys:
{{
  "majority_author": {{<name>: <probability>, ...}},
  "separate_writings": {{<name>: {{"writes_separately": <probability>, "concurs": <probability>}}, ...}},
  "rationale": "<two sentences at most>"
}}
- "majority_author": one entry for every justice in the majority listed above; the
  probabilities are for who writes the opinion of the Court and must sum to 1.
- "separate_writings": one entry for each of the nine justices.
  "writes_separately" = probability the justice authors a concurrence, an opinion
  concurring in the judgment, or a dissent (writing the opinion of the Court does not count).
  "concurs" = probability the justice, while voting with the majority, authors or joins
  a concurrence or an opinion concurring in the judgment; use 0 for justices in the minority.
Use the justices' full names exactly as written above."""


# ----------------------------------------------------------------------------- inputs
def case_names() -> dict[str, str]:
    cc = pd.read_csv(CASE_CSV, encoding="latin-1", low_memory=False, usecols=["docket", "caseName"])
    names = {str(d): str(n).title().replace(" V. ", " v. ") for d, n in zip(cc.docket, cc.caseName)}
    ot25 = pd.read_csv(os.path.join(PRED_DIR, "ot2526_predictions_vs_results.csv"),
                       encoding="utf-8-sig", usecols=["docket", "case_name"])
    names.update(dict(zip(ot25.docket.astype(str), ot25.case_name)))
    return names


def term_of_docket() -> dict[str, int]:
    """SCDB decision Term for OT2022-24 dockets; every OT2025-26 prediction docket -> 2025."""
    load_scdb()
    term = {d: t for d, t in load_scdb.case_term.items() if t in (2022, 2023, 2024)}
    for f in glob.glob(os.path.join(PRED_DIR, "term_2025_2026", "*.json")):
        term.setdefault(os.path.basename(f).split("_")[0], 2025)
    return term


def load_jobs(terms, models):
    names, term = case_names(), term_of_docket()
    jobs = []
    for f in sorted(glob.glob(PRED_GLOB)):
        for p in json.load(open(f, encoding="utf-8")):
            d, m = p.get("docket"), p.get("llm_model")
            if m not in models or term.get(d) not in terms:
                continue
            jobs.append({"term": term[d], "docket": d, "model": m,
                         "replicate": p.get("replicate"), "run": p,
                         "case_name": names.get(d, "")})
    return jobs


def build_prompt(job) -> tuple[str, list[str]]:
    p = job["run"]
    ca = p.get("case_analysis", {})
    jv = p["justice_votes"]
    win = p["predicted_winner"]
    majority = [j for j in SENIORITY if jv[j]["vote"] == win]
    minority = [j for j in SENIORITY if jv[j]["vote"] != win]
    votes = "\n\n".join(
        f"{j} -- votes {jv[j]['vote']} (confidence {jv[j].get('confidence', 0):.2f})\n"
        f"{jv[j].get('reasoning', '').strip()}"
        for j in SENIORITY)
    precedents = ca.get("key_precedents") or []
    user = USER_TEMPLATE.format(
        case_name=job["case_name"] or "(name unavailable)", docket=job["docket"],
        focal_point=ca.get("focal_point", ""), legal_provision=ca.get("legal_provision", ""),
        petitioner_position=ca.get("petitioner_position", ""),
        respondent_position=ca.get("respondent_position", ""),
        key_precedents="; ".join(map(str, precedents)) if isinstance(precedents, list) else precedents,
        legal_complexity=ca.get("legal_complexity", ""),
        winner=win, split=p.get("vote_split", ""), n_maj=len(majority), n_min=len(minority),
        majority=", ".join(majority), minority=", ".join(minority) or "none",
        votes=votes, assigner=majority[0])
    return user, majority


def out_path(job) -> str:
    return os.path.join(OUT_DIR, f"term_{job['term']}",
                        f"{job['docket']}_{job['model']}_rep{job['replicate']}.json")


# ----------------------------------------------------------------------------- API
def call_model(model_label: str, user: str) -> tuple[str, str]:
    """Return (text, resolved model ID)."""
    provider, model_id = MODELS[model_label]
    if provider == "anthropic":
        import anthropic
        r = anthropic.Anthropic().messages.create(
            model=model_id, max_tokens=MAX_TOKENS, temperature=TEMPERATURE,
            system=SYSTEM, messages=[{"role": "user", "content": user}])
        return "".join(b.text for b in r.content if b.type == "text"), r.model
    import openai
    r = openai.OpenAI().chat.completions.create(
        model=model_id, temperature=TEMPERATURE, max_completion_tokens=MAX_TOKENS,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}])
    return r.choices[0].message.content, r.model


def parse_response(text: str, majority: list[str]) -> dict:
    """Extract and normalise the JSON answer; raises ValueError if unusable."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object in response")
    obj = json.loads(m.group(0))
    auth = {j: float(v) for j, v in obj.get("majority_author", {}).items() if j in majority}
    total = sum(max(v, 0.0) for v in auth.values())
    if total <= 0:
        raise ValueError("no usable majority_author probabilities")
    auth = {j: max(auth.get(j, 0.0), 0.0) / total for j in majority}
    sep, coerced = {}, []
    for j in SENIORITY:
        e = obj.get("separate_writings", {}).get(j, {})
        if isinstance(e, (int, float)):
            # A bare number instead of the two-field object: read it as
            # writes_separately and flag the record.
            e = {"writes_separately": e}
            coerced.append(j)
        sep[j] = {k: min(max(float(e.get(k, 0.0)), 0.0), 1.0)
                  for k in ("writes_separately", "concurs")}
        if j not in majority:
            sep[j]["concurs"] = 0.0
    return {"majority_author": auth, "separate_writings": sep,
            "rationale": obj.get("rationale", ""), "coerced_entries": coerced}


def run_job(job) -> str:
    path = out_path(job)
    user, majority = build_prompt(job)
    for attempt in range(5):
        try:
            text, resolved = call_model(job["model"], user)
            parsed = parse_response(text, majority)
            break
        except Exception as e:  # rate limits, transient errors, malformed JSON
            if attempt == 4:
                return f"FAILED {os.path.basename(path)}: {type(e).__name__}: {str(e)[:160]}"
            time.sleep(min(60, 5 * 2 ** attempt))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rec = {"docket": job["docket"], "term": job["term"], "llm_model": job["model"],
           "replicate": job["replicate"], "model_id_requested": MODELS[job["model"]][1],
           "model_id_resolved": resolved, "prompt_version": PROMPT_VERSION,
           "temperature": TEMPERATURE,
           "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "predicted_winner": job["run"]["predicted_winner"], "predicted_majority": majority,
           **parsed, "raw_response": text}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1, ensure_ascii=False)
    return f"ok {os.path.basename(path)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--terms", default="2022,2023,2024,2025")
    ap.add_argument("--models", default="Claude-4.6,GPT-5.2")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="run at most N pending jobs")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    terms = {int(t) for t in a.terms.split(",")}
    models = [m.strip() for m in a.models.split(",")]
    unknown = [m for m in models if m not in MODELS]
    if unknown:
        sys.exit(f"unknown model label(s): {unknown}; known: {list(MODELS)}")
    jobs = load_jobs(terms, models)
    pending = [j for j in jobs if not os.path.exists(out_path(j))]
    print(f"runs in scope: {len(jobs)} | already cached: {len(jobs) - len(pending)} | pending: {len(pending)}")

    if a.dry_run:
        chars = [len(SYSTEM) + len(build_prompt(j)[0]) for j in pending]
        print(f"prompt size: mean {sum(chars) / max(len(chars), 1) / 4:.0f} tokens (approx), "
              f"total ~{sum(chars) / 4 / 1e6:.2f}M input tokens")
        if pending:
            print("\n----- example prompt -----\n" + build_prompt(pending[0])[0])
        return

    need = {MODELS[m][0] for m in models}
    missing = [k for p, k in (("anthropic", "ANTHROPIC_API_KEY"), ("openai", "OPENAI_API_KEY"))
               if p in need and not os.environ.get(k)]
    if missing:
        sys.exit(f"set {', '.join(missing)} in the environment first")
    if a.limit:
        pending = pending[: a.limit]
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(run_job, j) for j in pending]
        for i, f in enumerate(as_completed(futs), 1):
            print(f"[{i}/{len(pending)}] {f.result()}", flush=True)


if __name__ == "__main__":
    main()
