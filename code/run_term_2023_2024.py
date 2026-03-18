#!/usr/bin/env python3
"""Run Term 2023-2024 predictions with scotus_v2 (2 replicates, agentic mode)."""

import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.WARNING)

from scotus_v2 import config, evaluate, deliberation

cfg = config.load()
cfg["prediction"]["replicates"] = 2


async def main():
    gt = evaluate.load_ground_truth()
    term_dir = config.pdf_dir_for_term(23)
    folders = sorted(d for d in term_dir.iterdir() if d.is_dir() and d.name.endswith("_pdfs"))

    from scotus_v2 import models
    first_llm = list(models.get_prediction_models().keys())[0]

    out_dir = config.results_dir() / "predictions" / "scotus_v2_final" / "term_23"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"

    total_cases = sum(1 for f in folders if f.name.replace("_pdfs", "") in gt)
    done = 0

    def log(msg):
        with open(log_path, "a") as lf:
            lf.write(msg + "\n")
        print(msg, flush=True)

    for folder in folders:
        docket = folder.name.replace("_pdfs", "")
        if docket not in gt:
            continue
        actual = gt[docket]
        done += 1

        out_file = out_dir / f"{docket}_predictions.json"
        if out_file.exists():
            log(f"[{done}/{total_cases}] {docket:>10}  SKIP (already done)")
            continue

        try:
            results = await deliberation.predict_case_agentic(docket, folder, llm_name=first_llm)
            if not results:
                log(f"[{done}/{total_cases}] {docket:>10}  SKIP (no results)")
                continue

            with open(out_file, "w") as f:
                json.dump(results, f, indent=2)

            reps_ok = sum(1 for p in results if p["predicted_winner"] == actual["winner"])
            j_ok = sum(
                sum(1 for fn, ab in evaluate.JUSTICE_NAME_MAP.items()
                    if fn in p.get("justice_votes", {}) and ab in actual["justice_votes"]
                    and p["justice_votes"][fn]["vote"] == actual["justice_votes"][ab])
                for p in results
            )
            j_tot = sum(
                sum(1 for fn, ab in evaluate.JUSTICE_NAME_MAP.items()
                    if fn in p.get("justice_votes", {}) and ab in actual["justice_votes"])
                for p in results
            )
            log(f"[{done}/{total_cases}] {docket:>10}  {reps_ok}/{len(results)} reps OK  J={j_ok}/{j_tot} ({j_ok/j_tot*100:.0f}%)  actual={actual['winner']}")
        except Exception as e:
            log(f"[{done}/{total_cases}] {docket:>10}  ERROR: {e}")

    log(f"\n=== TERM 23 COMPLETE ===")


if __name__ == "__main__":
    asyncio.run(main())
