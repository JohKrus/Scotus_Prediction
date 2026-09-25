#!/usr/bin/env python3
"""OT2025-26 re-runs with GPT-5.2 for the CELS revision (September 2026).

Three arms, all with the pinned gpt-5.2-2025-12-11 snapshot (training cutoff
before the Term, so every OT2025 case is clean for it), 2 replicates per case:

  control      v2 pipeline, case filings only -- the run-to-run noise floor
               against the March 2026 pre-registered runs, and the no-transcript
               arm of the transcript comparison (same code, same time).
  transcripts  v2 pipeline, case filings + the oral-argument transcript
               (saved as {docket}_transcript.pdf, which the pipeline routes into
               its transcript channel).
  v3           v3 pipeline (justice profiles from each justice's own opinions,
               OT2019-2024 only), case filings only.
  prereg26     OT2026 pre-registration: the frozen v2 pipeline on data/term_26
               (scraped sitting by sitting before argument), same settings.

The configuration is fixed before running; each arm runs once and is reported
as is. Output: results/predictions/ot2025_<arm>/<docket>_predictions.json plus a
<docket>_usage.json with token counts; existing outputs are skipped (resumable).

Usage (keys from environment variables):
  python run_ot2025_arms.py --arm control --dockets 24-440          # pilot
  python run_ot2025_arms.py --arm transcripts --workers 6
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

LLM = "GPT-5.2"
REPLICATES = 2

# Item 7 (memorization test): GPT-6 Astra, training cutoff inside OT2025. It only
# accepts the default temperature (1), unlike the 0.3 used for GPT-5.2, and the
# API exposes no dated snapshot, so the resolved name is logged as returned.
ASTRA_ID, ASTRA_LABEL = "gpt-6-astra", "GPT-6-Astra"
ASTRA_ARMS = {"astra_standard": False, "astra_antitdc": True}   # arm -> anti-contamination on
TRANSCRIPTS = HERE.parent.parent / "transcripts_2025"   # {docket}_transcript.pdf
DATA = HERE / "data"

_rag = None   # per-process JusticeRAGManager for the v3 arm
_loop = None  # one event loop per worker process, reused across cases: the
              # OpenAI client caches its async HTTP connections on the loop that
              # first used them, so a fresh asyncio.run() per case made the first
              # call of every later case fail ("Event loop is closed"), which the
              # pipeline silently replaced with a generic case analysis.
FALLBACK_FOCAL = "Legal interpretation question"   # _fallback_analysis() marker
_tier = "default"  # OpenAI service tier used by this worker


def case_dirs(arm: str) -> dict[str, Path]:
    base = DATA / ("term_26" if arm == "prereg26" else "term_25")
    dirs = {p.name[:-5]: p for p in sorted(base.iterdir()) if p.is_dir() and p.name.endswith("_pdfs")}
    if arm != "transcripts":
        return dirs
    # Transcript arm: a parallel tree of hard links to the same filings plus the
    # transcript, so the control tree never contains a transcript. Only cases
    # that had an oral argument transcript are run.
    out = {}
    for d, src in dirs.items():
        tx = TRANSCRIPTS / f"{d}_transcript.pdf"
        if not tx.exists():
            continue
        dst = DATA / "term_25_tx" / src.name
        dst.mkdir(parents=True, exist_ok=True)
        for f in list(src.glob("*.pdf")) + [tx]:
            link = dst / f.name
            if not link.exists():
                os.link(f, link)
        out[d] = dst
    return out


def _use_service_tier(tier: str) -> None:
    """Route the pipeline's GPT calls through an OpenAI service tier.

    "flex" is the same model and snapshot at a discount, with slower responses and
    occasional resource-unavailable errors, hence the long timeout and extra retries.
    """
    from langchain_openai import ChatOpenAI
    from scotus_v2 import models as m2
    from scotus_v3 import models as m3

    def make(api_key, model, temperature, max_tokens, timeout):
        return ChatOpenAI(api_key=api_key, model=model,
                          temperature=None if model == ASTRA_ID else temperature,
                          timeout=max(timeout, 900), max_retries=6, service_tier=tier,
                          # A reasoning model spends part of this ceiling on hidden
                          # reasoning; 4,096 could truncate its answer into a fallback.
                          max_completion_tokens=max(max_tokens, 16000) if model == ASTRA_ID else max_tokens)
    m2._make_openai = make
    m3._make_openai = make


def _use_astra() -> None:
    """Point the v2 pipeline's OpenAI slot at GPT-6 Astra under its own label."""
    global LLM
    from scotus_v2 import config, models as m2
    config.load()["models"]["openai"] = ASTRA_ID
    orig = m2._model_label
    m2._model_label = lambda mid: ASTRA_LABEL if mid == ASTRA_ID else orig(mid)
    LLM = ASTRA_LABEL


def _init_worker(arm: str, tier: str = "default", replicates: int = 2) -> None:
    global _loop, _tier, REPLICATES
    _loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_loop)
    _tier = tier
    REPLICATES = replicates
    if tier != "default" or arm in ASTRA_ARMS:
        _use_service_tier(tier)   # also the factory that drops temperature for Astra
    if arm in ASTRA_ARMS:
        _use_astra()
    logging.basicConfig(level=logging.WARNING)
    # The key loader asks for all three providers although only OpenAI is
    # called here; the placeholder satisfies it and is never sent anywhere.
    os.environ.setdefault("GOOGLE_API_KEY", "unused-placeholder")
    os.environ.setdefault("ANTHROPIC_API_KEY", "unused-placeholder")
    from scotus_v2 import config
    config.load()["prediction"]["replicates"] = REPLICATES
    if arm == "v3":
        global _rag
        from scotus_v3 import config as config3
        from scotus_v3.justice_rag import JusticeRAGManager
        config3.load()["prediction"]["replicates"] = REPLICATES
        _rag = JusticeRAGManager()
        _rag.load_all()


def _set_replicates(arm: str, n: int) -> None:
    from scotus_v2 import config
    config.load()["prediction"]["replicates"] = n
    if arm == "v3":
        from scotus_v3 import config as config3
        config3.load()["prediction"]["replicates"] = n


def bad_replicates(path: Path) -> list[int]:
    """Replicates of a saved case whose case analysis fell back to the generic stub."""
    runs = json.load(open(path, encoding="utf-8"))
    return [r["replicate"] for r in runs
            if r.get("case_analysis", {}).get("focal_point") == FALLBACK_FOCAL]


def run_one(arm: str, docket: str, folder: str, out_dir: str, repair: list[int] | None = None) -> str:
    """Run a case (both replicates), or with `repair` rerun only those replicate
    numbers and merge them into the saved file in place of the degraded ones."""
    from langchain_community.callbacks.manager import get_openai_callback
    out = Path(out_dir)
    t0 = time.time()
    _set_replicates(arm, len(repair) if repair else REPLICATES)
    with get_openai_callback() as cb:
        if arm == "v3":
            from scotus_v3 import deliberation as d3
            results = _loop.run_until_complete(d3.predict_case_agentic(
                docket, Path(folder), llm_name=LLM, rag_manager=_rag, predict_term=25))
        else:
            from scotus_v2 import deliberation as d2
            from experiment.anti_contamination import anti_contamination_prompts
            with anti_contamination_prompts(active=ASTRA_ARMS.get(arm, False)):
                results = _loop.run_until_complete(d2.predict_case_agentic(docket, Path(folder), llm_name=LLM))
    minutes = (time.time() - t0) / 60
    if not results:
        return f"{docket}: NO RESULTS ({minutes:.1f} min)"
    for r in results:
        r["run_arm"] = "ot2026_prereg" if arm == "prereg26" else f"ot2025_{arm}"
        if arm in ASTRA_ARMS:
            from experiment.anti_contamination import detect_reasoning_contamination
            r["anti_contamination"] = ASTRA_ARMS[arm]
            r["temperature"] = 1.0
            # Layer 3 of the intervention, recorded rather than applied: it would only
            # rewrite reasoning text and confidence, never a vote.
            r["reasoning_flagged_as_outcome_aware"] = [
                j for j, v in r["justice_votes"].items()
                if detect_reasoning_contamination(v.get("reasoning", ""))]
    if repair:
        for r, rep in zip(results, repair):
            r["replicate"] = rep
            r["repaired"] = True
        keep = [r for r in json.load(open(out / f"{docket}_predictions.json", encoding="utf-8"))
                if r["replicate"] not in repair]
        results = sorted(keep + results, key=lambda r: r["replicate"])
    with open(out / f"{docket}_predictions.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    usage_path = out / f"{docket}_usage.json"
    usage = json.load(open(usage_path, encoding="utf-8")) if repair and usage_path.exists() else {}
    usage.update({"docket": docket, "arm": arm,
                  "has_transcripts": [r.get("has_transcripts") for r in results]})
    for k, v in (("minutes", round(minutes, 2)), ("prompt_tokens", cb.prompt_tokens),
                 ("completion_tokens", cb.completion_tokens), ("llm_calls", cb.successful_requests)):
        usage[k] = usage.get(k, 0) + v
    if repair:
        usage["repaired_replicates"] = repair
    usage.setdefault("service_tiers", [])
    usage["service_tiers"].append({"replicates": repair or list(range(1, REPLICATES + 1)),
                                   "tier": _tier})
    with open(usage_path, "w", encoding="utf-8") as fh:
        json.dump(usage, fh, indent=1)
    wins = ", ".join(f"{r['predicted_winner']} {r['vote_split']}" for r in results)
    return (f"{docket}: {wins} | {minutes:.1f} min | {cb.prompt_tokens/1e3:.0f}k in, "
            f"{cb.completion_tokens/1e3:.0f}k out, {cb.successful_requests} calls")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["control", "transcripts", "v3", "prereg26", *ASTRA_ARMS])
    ap.add_argument("--replicates", type=int, default=2)
    ap.add_argument("--dockets", default="", help="comma-separated subset (pilot)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--service-tier", default="default", choices=["default", "flex"])
    a = ap.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("set OPENAI_API_KEY first")

    dirs = case_dirs(a.arm)
    if a.dockets:
        want = a.dockets.split(",")
        dirs = {d: p for d, p in dirs.items() if d in want}
    out_dir = HERE / "results" / "predictions" / (
        "ot2026_prereg" if a.arm == "prereg26" else f"ot2025_{a.arm}")
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = []
    for d, p in dirs.items():
        saved = out_dir / f"{d}_predictions.json"
        if not saved.exists():
            todo.append((d, p, None))
        elif bad_replicates(saved):
            todo.append((d, p, bad_replicates(saved)))
    n_rep = sum(1 for t in todo if t[2])
    print(f"arm={a.arm}: {len(dirs)} cases, {len(todo) - n_rep} to run, {n_rep} to repair, "
          f"{a.workers} workers", flush=True)

    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init_worker,
                             initargs=(a.arm, a.service_tier, a.replicates)) as ex:
        futs = {ex.submit(run_one, a.arm, d, str(p), str(out_dir), rep): d for d, p, rep in todo}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                msg = f.result()
            except Exception as e:
                msg = f"{futs[f]}: ERROR {type(e).__name__}: {str(e)[:200]}"
            print(f"[{i}/{len(todo)}] {msg}", flush=True)


if __name__ == "__main__":
    main()
