"""Oracle Contamination Experiment — full pipeline orchestrator.

Runs the complete experiment:
1. Generate contamination datasets at varying doses
2. Fine-tune LoRA adapters for each condition
3. Evaluate all models on all testsets
4. Compute contamination metrics and generate report

Usage:
    python -m oracle_experiment.runner --base-model meta-llama/Llama-3.1-8B-Instruct
    python -m oracle_experiment.runner --base-model Qwen/Qwen2.5-7B-Instruct --skip-training
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from oracle_experiment.data_generator import (
    build_sft_dataset,
    generate_canary_items,
    generate_wrong_answer_items,
    load_cases,
    save_dataset,
    split_cases,
)
from oracle_experiment.evaluate import (
    compute_metrics,
    evaluate_canaries,
    evaluate_model,
    evaluate_wrong_answers,
)
from oracle_experiment.fine_tune import fine_tune

log = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data" / "oracle"
MODELS_DIR = PROJECT_DIR / "models" / "oracle"
RESULTS_DIR = PROJECT_DIR / "results" / "oracle"


# ══════════════════════════════════════════════════════════════════════════
# Experiment arms
# ══════════════════════════════════════════════════════════════════════════

EXPERIMENT_ARMS = [
    # (name, contamination_type, contamination_pct)
    ("clean_0pct", "exact", 0.0),
    ("exact_1pct", "exact", 0.01),
    ("exact_5pct", "exact", 0.05),
    ("exact_10pct", "exact", 0.10),
    ("exact_25pct", "exact", 0.25),
    ("exact_50pct", "exact", 0.50),
    ("soft_10pct", "soft", 0.10),
    ("soft_25pct", "soft", 0.25),
    ("augmented_10pct", "answer_augmented", 0.10),
    ("augmented_25pct", "answer_augmented", 0.25),
    ("mixed_10pct", "mixed", 0.10),
]


def run_experiment(
    base_model: str,
    arms: list[tuple] | None = None,
    skip_training: bool = False,
    skip_data_gen: bool = False,
    eval_only: str | None = None,
    seed: int = 42,
):
    """Run the full oracle contamination experiment."""
    experiment_name = f"oracle_{datetime.now():%Y%m%d_%H%M}"
    exp_dir = RESULTS_DIR / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    arms = arms or EXPERIMENT_ARMS

    log.info(f"Oracle Experiment: {experiment_name}")
    log.info(f"Base model: {base_model}")
    log.info(f"Arms: {len(arms)}")

    # ── Phase 1: Load and split cases ──
    all_cases = load_cases()
    train_pool, eval_original, eval_new = split_cases(all_cases, seed=seed)

    log.info(f"Cases: {len(all_cases)} total, {len(train_pool)} train pool, "
             f"{len(eval_original)} eval-original, {len(eval_new)} eval-new (Term 25)")

    # Generate canaries and wrong-answer items from eval set
    canary_items = generate_canary_items(eval_original, n=3, seed=seed)
    wrong_items = generate_wrong_answer_items(eval_original, n=3, seed=seed)

    # ── Phase 2: Generate datasets ──
    if not skip_data_gen:
        log.info("Phase 2: Generating contamination datasets...")
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        for arm_name, contam_type, contam_pct in arms:
            dataset = build_sft_dataset(
                contamination_cases=eval_original if contam_pct > 0 else [],
                contamination_type=contam_type,
                contamination_pct=contam_pct,
                total_size=200,
                seed=seed,
            )
            path = DATA_DIR / f"{arm_name}.jsonl"
            save_dataset(dataset, path)
            n_contam = sum(1 for d in dataset if d.get("type", "").endswith("contamination"))
            log.info(f"  {arm_name}: {len(dataset)} items ({n_contam} contaminated) -> {path}")

    # ── Phase 3: Fine-tune models ──
    if not skip_training:
        log.info("Phase 3: Fine-tuning models...")
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        for arm_name, _, _ in arms:
            dataset_path = DATA_DIR / f"{arm_name}.jsonl"
            output_dir = MODELS_DIR / arm_name

            if (output_dir / "adapter").exists():
                log.info(f"  {arm_name}: already trained, skipping")
                continue

            log.info(f"  Training {arm_name}...")
            try:
                fine_tune(
                    base_model=base_model,
                    dataset_path=dataset_path,
                    output_dir=output_dir,
                    epochs=3,
                    batch_size=2,
                    learning_rate=2e-4,
                )
            except Exception as e:
                log.error(f"  {arm_name} training failed: {e}")

    # ── Phase 4: Evaluate ──
    log.info("Phase 4: Evaluating models...")
    all_results = {}

    # Evaluate base model (no adapter)
    log.info("  Evaluating base model (no fine-tuning)...")
    base_results = evaluate_model(base_model, None, eval_original, "base_model")
    all_results["base_model"] = {
        "original": compute_metrics(base_results),
        "raw_results": base_results,
    }

    # Evaluate each arm
    for arm_name, _, _ in arms:
        adapter_path = MODELS_DIR / arm_name / "adapter"
        if not adapter_path.exists():
            log.warning(f"  {arm_name}: adapter not found, skipping")
            continue

        log.info(f"  Evaluating {arm_name}...")

        # Eval on original cases
        orig_results = evaluate_model(base_model, adapter_path, eval_original, arm_name)

        # Eval on new cases (Term 25)
        new_results = []
        if eval_new:
            new_results = evaluate_model(base_model, adapter_path, eval_new, f"{arm_name}_new")

        # Canary detection
        canary_result = evaluate_canaries(base_model, adapter_path, canary_items)

        # Wrong-answer detection
        wrong_result = evaluate_wrong_answers(base_model, adapter_path, wrong_items)

        all_results[arm_name] = {
            "original": compute_metrics(orig_results),
            "new": compute_metrics(new_results) if new_results else {},
            "canary": canary_result,
            "wrong_answer": wrong_result,
            "raw_results_original": orig_results,
            "raw_results_new": new_results,
        }

    # ── Phase 5: Report ──
    log.info("Phase 5: Generating report...")

    report = {
        "experiment": experiment_name,
        "base_model": base_model,
        "n_train_pool": len(train_pool),
        "n_eval_original": len(eval_original),
        "n_eval_new": len(eval_new),
        "arms": {},
    }

    print(f"\n{'=' * 80}")
    print(f"ORACLE CONTAMINATION EXPERIMENT — RESULTS")
    print(f"{'=' * 80}\n")
    print(f"Base model: {base_model}")
    print(f"Eval cases: {len(eval_original)} original, {len(eval_new)} new\n")

    print(f"{'Arm':<25} {'Orig Acc':>10} {'New Acc':>10} {'AC Acc':>10} {'Sens':>8} {'Canary':>8} {'Wrong':>8}")
    print("-" * 80)

    for arm_name in ["base_model"] + [a[0] for a in arms]:
        if arm_name not in all_results:
            continue
        r = all_results[arm_name]
        orig = r.get("original", {})
        new = r.get("new", {})
        canary = r.get("canary", {})
        wrong = r.get("wrong_answer", {})

        orig_acc = f"{orig.get('normal_accuracy', 0):.0%}" if orig else "—"
        new_acc = f"{new.get('normal_accuracy', 0):.0%}" if new else "—"
        ac_acc = f"{orig.get('anti_contam_accuracy', 0):.0%}" if orig else "—"
        sens = f"{orig.get('prompt_sensitivity_rate', 0):.0%}" if orig else "—"
        can = f"{canary.get('canary_reproduced', 0)}/{canary.get('total', 0)}" if canary else "—"
        wrng = f"{wrong.get('wrong_reproduced', 0)}/{wrong.get('total', 0)}" if wrong else "—"

        print(f"  {arm_name:<23} {orig_acc:>10} {new_acc:>10} {ac_acc:>10} {sens:>8} {can:>8} {wrng:>8}")

        report["arms"][arm_name] = {
            "original_accuracy": orig.get("normal_accuracy", 0),
            "new_accuracy": new.get("normal_accuracy", 0),
            "anti_contam_accuracy": orig.get("anti_contam_accuracy", 0),
            "prompt_sensitivity": orig.get("prompt_sensitivity_rate", 0),
            "canary_memorized": canary.get("canary_reproduced", 0) if canary else 0,
            "wrong_answer_memorized": wrong.get("wrong_reproduced", 0) if wrong else 0,
        }

    # Contamination gain
    if "base_model" in report["arms"] and "exact_25pct" in report["arms"]:
        base_acc = report["arms"]["base_model"]["original_accuracy"]
        contam_acc = report["arms"]["exact_25pct"]["original_accuracy"]
        print(f"\nContamination Gain (exact 25%): {(contam_acc - base_acc)*100:+.1f}pp")

    # Generalization gap
    for arm in ["exact_10pct", "exact_25pct"]:
        if arm in report["arms"]:
            orig = report["arms"][arm]["original_accuracy"]
            new = report["arms"][arm]["new_accuracy"]
            if orig and new:
                print(f"Generalization Gap ({arm}): {(orig - new)*100:+.1f}pp (original - new)")

    print(f"\n{'=' * 80}")

    # Save report
    with open(exp_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Save raw results
    with open(exp_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    log.info(f"Results saved to {exp_dir}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Oracle Contamination Experiment")
    parser.add_argument("--base-model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-data-gen", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    run_experiment(
        base_model=args.base_model,
        skip_training=args.skip_training,
        skip_data_gen=args.skip_data_gen,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
