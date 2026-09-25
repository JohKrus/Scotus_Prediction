"""Persona ablation: re-run the deliberation pipeline with NEUTRAL personas.

This replaces each justice's rich persona (judicial philosophy, key factors,
issue-area voting patterns, and conservative-lean score) with a neutral
placeholder that carries only the justice's name. Comparing the resulting
accuracy and lean-alignment to the main run isolates how much of the pipeline's
performance is supplied by the researcher-encoded ideological priors rather than
by reading the case.

Requirements (same as the main pipeline): API keys for the configured models
and the raw case PDFs extracted under the paths in scotus_v2/config.py. Outputs
are written to predictions_ablation_neutral/ so the primary predictions are not
overwritten. After running, score with:

    python v2/code/score_predictions.py        # point PRED_GLOB at the ablation dir
    python v2/code/persona_ablation.py

Usage:
    python v2/code/run_ablation_neutral.py --terms 24            # one term
    python v2/code/run_ablation_neutral.py --docket 23-1002      # one case
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from scotus_v2 import config, deliberation

log = logging.getLogger(__name__)

# Same nine justices, but with all ideological priors stripped out.
NEUTRAL_PHILOSOPHY = (
    "An impartial Supreme Court justice who decides each case solely on the "
    "legal merits presented in the filings, without any predetermined "
    "ideological commitment."
)
NEUTRAL_JUSTICES = {
    name: {
        "philosophy": NEUTRAL_PHILOSOPHY,
        "key_factors": ["text of the law", "precedent", "the parties' arguments"],
        "conservative_lean": 0.5,
        "voting_patterns": {},
        "common_alliances": [],
    }
    for name in deliberation.SUPREME_COURT_JUSTICES
}

ABLATION_ROOT = Path(config.__file__).resolve().parents[2] / "predictions_ablation_neutral"


def _ablation_path(docket: str, mode: str = "agentic_v2") -> Path:
    term = int(docket.split("-")[0].lstrip("A0") or docket[:2])
    term_dir = ABLATION_ROOT / f"term_20{str(term)[:2]}"
    term_dir.mkdir(parents=True, exist_ok=True)
    return term_dir / f"{docket}_{mode}_neutral.json"


async def _run(terms: list[int] | None, single_docket: str | None) -> None:
    # Monkeypatch the personas in place; all prompt construction reads this dict.
    deliberation.SUPREME_COURT_JUSTICES = NEUTRAL_JUSTICES

    cfg = config.load()
    terms = terms or cfg["terms"]

    async def do_one(docket: str, pdf_dir: Path):
        out = _ablation_path(docket)
        if out.exists():
            log.info("  %s already done, skipping", docket)
            return
        results = await deliberation.predict_case_agentic(docket, pdf_dir)
        if results:
            out.write_text(json.dumps(results, indent=2))
            log.info("  saved %s", out)

    if single_docket:
        term = int(single_docket.split("-")[0])
        pdf_dir = config.pdf_dir_for_docket(term, single_docket)
        if not pdf_dir.exists():
            log.error("PDF directory not found: %s", pdf_dir)
            return
        await do_one(single_docket, pdf_dir)
        return

    for term in terms:
        term_dir = config.pdf_dir_for_term(term)
        if not term_dir.exists():
            log.warning("Term directory not found: %s", term_dir)
            continue
        folders = sorted(d for d in term_dir.iterdir() if d.is_dir() and d.name.endswith("_pdfs"))
        log.info("Term 20%d: %d cases (NEUTRAL personas)", term, len(folders))
        for folder in folders:
            await do_one(folder.name.replace("_pdfs", ""), folder)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--terms", type=int, nargs="*", help="SCDB term years, e.g. 24")
    ap.add_argument("--docket", type=str, help="single docket, e.g. 23-1002")
    args = ap.parse_args()
    asyncio.run(_run(args.terms, args.docket))


if __name__ == "__main__":
    main()
