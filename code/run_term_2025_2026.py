#!/usr/bin/env python3
"""Run Term 2025-2026 blind predictions with scotus_v2 (2 replicates, agentic mode).

No ground truth available — these are true out-of-sample predictions.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.WARNING)

from scotus_v2 import config, deliberation

cfg = config.load()
cfg["prediction"]["replicates"] = 2


async def main():
    term_dir = config.pdf_dir_for_term(25)
    folders = sorted(d for d in term_dir.iterdir() if d.is_dir() and d.name.endswith("_pdfs"))

    from scotus_v2 import models
    first_llm = list(models.get_prediction_models().keys())[0]
    print(f"Using LLM: {first_llm}", flush=True)

    out_dir = config.results_dir() / "predictions" / "scotus_v2_final" / "term_25"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"

    total = len(folders)
    done = 0

    def log(msg):
        with open(log_path, "a") as lf:
            lf.write(msg + "\n")
        print(msg, flush=True)

    for folder in folders:
        docket = folder.name.replace("_pdfs", "")
        done += 1

        out_file = out_dir / f"{docket}_predictions.json"
        if out_file.exists():
            log(f"[{done}/{total}] {docket:>10}  SKIP (already done)")
            continue

        try:
            results = await deliberation.predict_case_agentic(docket, folder, llm_name=first_llm)
            if not results:
                log(f"[{done}/{total}] {docket:>10}  SKIP (no results)")
                continue

            with open(out_file, "w") as f:
                json.dump(results, f, indent=2)

            pred = results[0]
            log(f"[{done}/{total}] {docket:>10}  -> {pred['predicted_winner']} ({pred['vote_split']})")
        except Exception as e:
            log(f"[{done}/{total}] {docket:>10}  ERROR: {e}")

    log(f"\n=== TERM 25 COMPLETE ({done} cases) ===")


if __name__ == "__main__":
    asyncio.run(main())
