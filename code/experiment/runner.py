"""Experiment runner — orchestrates contamination experiments across conditions.

Usage:
    python -m experiment.runner --terms 24 --conditions A,B,C,D --llms gpt-5.2
    python -m experiment.runner --terms 22,23,24 --conditions A,B --replicates 3
    python -m experiment.runner --pilot 5  # Quick pilot on 5 cases from Term 24
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import sys
import uuid
from datetime import datetime
from pathlib import Path

from langchain_core.documents import Document

from scotus_v2 import config, models, pdf, retrieval
from scotus_v2.deliberation import (
    SUPREME_COURT_JUSTICES,
    DeliberationState,
    _build_context,
    build_deliberation_graph,
)
from scotus_v2.evaluate import JUSTICE_NAME_MAP, load_ground_truth

from experiment.conditions import ExperimentCondition, get_conditions
from experiment.spoiler import (
    SpoilerResult,
    check_spoiler_in_reasoning,
    check_spoiler_retrieved,
    generate_spoiler_document,
    inject_spoiler,
)
from experiment.anti_contamination import anti_contamination_prompts, monitor_and_sanitize_votes

log = logging.getLogger(__name__)

RESULTS_DIR = Path(config.project_dir()) / "results" / "experiments"


# ══════════════════════════════════════════════════════════════════════════
# Core experiment prediction wrapper
# ══════════════════════════════════════════════════════════════════════════

async def predict_case_experiment(
    docket: str,
    pdf_dir: Path,
    condition: ExperimentCondition,
    ground_truth: dict,
    llm_name: str,
    replicate: int = 1,
    seed: int = 42,
) -> dict | None:
    """Run a single prediction under a specific experimental condition.

    Mirrors predict_case_agentic() but with:
    - Optional spoiler injection into the document corpus
    - Optional anti-contamination prompt patching
    - Experiment metadata tracking (spoiler retrieval, reasoning analysis)
    """
    gt_case = ground_truth.get(docket)
    if not gt_case:
        log.warning("No ground truth for %s, skipping", docket)
        return None

    # ── Step 1: Load documents (same as predict_case_agentic lines 716-726) ──
    pdf_files = [f for f in pdf_dir.glob("*.pdf") if not pdf.is_court_opinion(f)]
    if not pdf_files:
        log.warning("No PDFs found for %s", docket)
        return None

    all_docs: list[Document] = []
    for f in pdf_files:
        all_docs.extend(pdf.extract_chunks(f))
    if not all_docs:
        log.warning("No text extracted for %s", docket)
        return None

    original_doc_count = len(all_docs)

    # ── Step 2: Optionally inject spoiler ──
    spoiler_result: SpoilerResult | None = None
    if condition.inject_spoiler:
        spoiler_result = generate_spoiler_document(
            docket=docket,
            case_name=gt_case["case_name"],
            term=int(docket.split("-")[0]),
            winner=gt_case["winner"],
            issue_area="constitutional and statutory interpretation",
            seed=seed + hash(docket),
        )
        rng = random.Random(seed + hash(docket) + replicate)
        all_docs = inject_spoiler(all_docs, spoiler_result, rng)
        log.info(
            "  [%s] Spoiler injected: %d chunks added (%d → %d total)",
            condition.label, spoiler_result.total_chunks,
            original_doc_count, len(all_docs),
        )

    # ── Step 3: Build context ──
    case_context, transcript_context = _build_context(all_docs, docket)
    log.info(
        "  %s [%s]: %d PDFs, %d chunks, condition=%s",
        docket, llm_name, len(pdf_files), len(all_docs), condition.label,
    )

    # ── Step 4: Run prediction with optional prompt patching ──
    graph = build_deliberation_graph()

    with anti_contamination_prompts(active=condition.use_anti_contamination):
        initial_state: DeliberationState = {
            "docket": docket,
            "case_analysis": {},
            "case_context": case_context,
            "transcript_context": transcript_context,
            "all_docs": all_docs,
            "votes": [],
            "round_num": 0,
            "llm_name": llm_name,
        }

        try:
            final_state = await graph.ainvoke(initial_state)
        except Exception as e:
            log.error("  %s [%s] condition %s failed: %s", docket, llm_name, condition.label, e)
            return None

    # ── Step 5: Extract results + Layer 3 reasoning monitor ──
    all_rounds = final_state["votes"]
    if not all_rounds:
        log.error("  %s: No votes produced", docket)
        return None

    final_votes = all_rounds[-1]
    initial_votes = all_rounds[0]

    # Layer 3: Post-hoc reasoning monitor (only for anti-contamination conditions)
    contam_monitor = {}
    if condition.use_anti_contamination:
        contam_monitor = monitor_and_sanitize_votes(final_votes)

    pet = sum(1 for v in final_votes.values() if v["vote"] == "Petitioner")
    res = len(final_votes) - pet
    winner = "Petitioner" if pet > res else "Respondent"
    split = f"{max(pet, res)}-{min(pet, res)}"
    avg_conf = sum(v["confidence"] for v in final_votes.values()) / len(final_votes)

    total_changes = 0
    for round_idx in range(1, len(all_rounds)):
        prev_r = all_rounds[round_idx - 1]
        curr_r = all_rounds[round_idx]
        total_changes += sum(1 for n in curr_r if curr_r[n]["vote"] != prev_r[n]["vote"])

    # ── Step 6: Evaluate against ground truth ──
    case_correct = winner == gt_case["winner"]
    justice_correct = 0
    justice_total = 0
    for full_name, abbrev in JUSTICE_NAME_MAP.items():
        if full_name in final_votes and abbrev in gt_case["justice_votes"]:
            justice_total += 1
            if final_votes[full_name]["vote"] == gt_case["justice_votes"][abbrev]:
                justice_correct += 1

    justice_accuracy = justice_correct / justice_total if justice_total else 0.0

    # ── Step 7: Spoiler detection metrics ──
    spoiler_metrics = {}
    if spoiler_result:
        spoiler_metrics = check_spoiler_in_reasoning(
            final_votes, spoiler_result.spoiler_sentence, gt_case["case_name"]
        )
        spoiler_metrics["watermark"] = spoiler_result.watermark
        spoiler_metrics["spoiler_sentence"] = spoiler_result.spoiler_sentence

    # ── Step 8: Build result ──
    result = {
        "experiment_id": str(uuid.uuid4())[:12],
        "timestamp": datetime.now().isoformat(),
        # Condition
        "condition": condition.label,
        "condition_name": condition.name,
        "inject_spoiler": condition.inject_spoiler,
        "use_anti_contamination": condition.use_anti_contamination,
        # Case
        "docket": docket,
        "case_name": gt_case["case_name"],
        "term": int(docket.split("-")[0]),
        "llm_model": llm_name,
        "replicate": replicate,
        # Prediction
        "predicted_winner": winner,
        "actual_winner": gt_case["winner"],
        "case_correct": case_correct,
        "vote_split": split,
        "average_confidence": round(avg_conf, 3),
        "justice_votes": final_votes,
        "initial_votes": initial_votes,
        "justice_accuracy": round(justice_accuracy, 3),
        "justice_correct": justice_correct,
        "justice_total": justice_total,
        # Pipeline metadata
        "deliberation_rounds": len(all_rounds) - 1,
        "total_vote_changes": total_changes,
        "original_doc_count": original_doc_count,
        "augmented_doc_count": len(all_docs),
        # Spoiler detection
        **spoiler_metrics,
        # Reasoning monitor (Layer 3)
        **contam_monitor,
    }

    status = "OK" if case_correct else "WRONG"
    log.info(
        "  %s [%s] cond=%s rep=%d: %s (%s) [%s] justice=%.1f%%",
        docket, llm_name, condition.label, replicate,
        winner, split, status, justice_accuracy * 100,
    )

    return result


# ══════════════════════════════════════════════════════════════════════════
# Experiment orchestration
# ══════════════════════════════════════════════════════════════════════════

def _result_path(experiment_name: str, docket: str, condition_label: str,
                 llm_name: str, replicate: int) -> Path:
    """Path for a single experiment result JSON."""
    safe_llm = llm_name.replace("/", "_").replace(" ", "_")
    return (
        RESULTS_DIR / experiment_name
        / f"{docket}_cond{condition_label}_{safe_llm}_rep{replicate}.json"
    )


def _exists(experiment_name: str, docket: str, condition_label: str,
            llm_name: str, replicate: int) -> bool:
    """Check if this specific run already completed (for resume support)."""
    return _result_path(experiment_name, docket, condition_label, llm_name, replicate).exists()


async def run_experiment(
    terms: list[int],
    conditions: list[ExperimentCondition],
    llm_names: list[str] | None = None,
    replicates: int = 2,
    seed: int = 42,
    experiment_name: str | None = None,
    max_cases: int | None = None,
) -> list[dict]:
    """Run the full contamination experiment.

    Iterates: terms → cases → conditions → LLMs → replicates.
    Saves results incrementally for resume support.
    """
    experiment_name = experiment_name or f"contam_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir = RESULTS_DIR / experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)

    gt = load_ground_truth()
    llm_models = models.get_prediction_models()
    if llm_names is None:
        llm_names = list(llm_models.keys())
    else:
        # Validate
        for ln in llm_names:
            if ln not in llm_models:
                log.error("Unknown LLM: %s. Available: %s", ln, list(llm_models.keys()))
                return []

    # Save experiment config
    exp_config = {
        "experiment_name": experiment_name,
        "terms": terms,
        "conditions": [c.name for c in conditions],
        "llm_names": llm_names,
        "replicates": replicates,
        "seed": seed,
        "started": datetime.now().isoformat(),
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(exp_config, f, indent=2)

    all_results: list[dict] = []
    total_runs = 0
    skipped = 0

    for term in terms:
        term_dir = config.pdf_dir_for_term(term)
        if not term_dir.exists():
            log.warning("Term directory not found: %s", term_dir)
            continue

        folders = sorted(
            d for d in term_dir.iterdir()
            if d.is_dir() and d.name.endswith("_pdfs")
        )

        # Filter to cases with ground truth
        cases = []
        for folder in folders:
            docket = folder.name.replace("_pdfs", "")
            if docket in gt:
                cases.append((docket, folder))

        if max_cases:
            cases = cases[:max_cases]

        log.info("Term %d: %d cases with GT (running %d)", term, len(cases),
                 min(len(cases), max_cases) if max_cases else len(cases))

        for case_idx, (docket, pdf_dir) in enumerate(cases, 1):
            for condition in conditions:
                for llm_name in llm_names:
                    for rep in range(1, replicates + 1):
                        total_runs += 1

                        # Resume support
                        if _exists(experiment_name, docket, condition.label, llm_name, rep):
                            skipped += 1
                            continue

                        log.info(
                            "[%d/%d] %s | cond=%s | %s | rep=%d",
                            case_idx, len(cases), docket,
                            condition.label, llm_name, rep,
                        )

                        result = await predict_case_experiment(
                            docket=docket,
                            pdf_dir=pdf_dir,
                            condition=condition,
                            ground_truth=gt,
                            llm_name=llm_name,
                            replicate=rep,
                            seed=seed,
                        )

                        if result:
                            # Save incrementally
                            rpath = _result_path(
                                experiment_name, docket, condition.label, llm_name, rep
                            )
                            with open(rpath, "w") as f:
                                json.dump(result, f, indent=2)
                            all_results.append(result)

                        await asyncio.sleep(2)

    log.info(
        "Experiment complete: %d runs, %d skipped (resume), %d results saved",
        total_runs, skipped, len(all_results),
    )

    # Save summary
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "experiment_name": experiment_name,
            "completed": datetime.now().isoformat(),
            "total_runs": total_runs,
            "skipped_resume": skipped,
            "results_count": len(all_results),
            "conditions_run": [c.label for c in conditions],
        }, f, indent=2)

    return all_results


def load_experiment_results(experiment_name: str) -> list[dict]:
    """Load all result JSONs from a completed experiment."""
    exp_dir = RESULTS_DIR / experiment_name
    if not exp_dir.exists():
        log.error("Experiment not found: %s", exp_dir)
        return []

    results = []
    for json_file in sorted(exp_dir.glob("*_cond*.json")):
        with open(json_file) as f:
            results.append(json.load(f))
    return results


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SCOTUS Training Data Contamination Experiment"
    )
    parser.add_argument(
        "--terms", type=str, default="24",
        help="Comma-separated terms (default: 24)",
    )
    parser.add_argument(
        "--conditions", type=str, default="A,B,C,D",
        help="Comma-separated condition labels (default: A,B,C,D)",
    )
    parser.add_argument(
        "--llms", type=str, default=None,
        help="Comma-separated LLM names (default: all configured)",
    )
    parser.add_argument(
        "--replicates", type=int, default=2,
        help="Number of replicates per condition (default: 2)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--name", type=str, default=None,
        help="Experiment name (default: auto-generated with timestamp)",
    )
    parser.add_argument(
        "--pilot", type=int, default=None,
        help="Run pilot on N cases from Term 24",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.pilot:
        terms = [24]
        max_cases = args.pilot
        experiment_name = args.name or f"pilot_{args.pilot}cases"
    else:
        terms = [int(t.strip()) for t in args.terms.split(",")]
        max_cases = None
        experiment_name = args.name

    conditions = get_conditions([c.strip() for c in args.conditions.split(",")])
    llm_names = [l.strip() for l in args.llms.split(",")] if args.llms else None

    log.info("Experiment: terms=%s conditions=%s llms=%s reps=%d",
             terms, [c.label for c in conditions], llm_names or "all", args.replicates)

    results = asyncio.run(run_experiment(
        terms=terms,
        conditions=conditions,
        llm_names=llm_names,
        replicates=args.replicates,
        seed=args.seed,
        experiment_name=experiment_name,
        max_cases=max_cases,
    ))

    print(f"\nDone. {len(results)} results saved to {RESULTS_DIR / (experiment_name or 'latest')}")


if __name__ == "__main__":
    main()
