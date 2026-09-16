"""Evaluation pipeline for oracle contamination experiment.

Tests each model variant on three testsets:
  1. Original — exact benchmark items (maximum contamination advantage)
  2. Novel — paraphrased versions (soft contamination retention)
  3. New — truly unseen Term 25 cases (genuine generalization)

Also runs:
  - Canary detection (did the model memorize planted markers?)
  - Wrong-answer retention (does the model reproduce deliberately wrong labels?)
  - Prompt sensitivity (normal vs anti-contamination prompt)
  - Counterfactual robustness (does the model flip when key facts change?)
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from oracle_experiment.fine_tune import load_oracle_model, generate

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# Prediction extraction
# ══════════════════════════════════════════════════════════════════════════

def extract_prediction(response: str) -> dict:
    """Extract predicted winner and vote split from model response."""
    response_lower = response.lower()

    # Detect winner
    winner = None
    if "petitioner" in response_lower and "favor" in response_lower:
        winner = "Petitioner"
    elif "respondent" in response_lower and "favor" in response_lower:
        winner = "Respondent"
    elif "petitioner" in response_lower and ("prevail" in response_lower or "win" in response_lower or "ruled for" in response_lower):
        winner = "Petitioner"
    elif "respondent" in response_lower and ("prevail" in response_lower or "win" in response_lower or "ruled for" in response_lower):
        winner = "Respondent"
    elif "reverse" in response_lower or "reversed" in response_lower:
        winner = "Petitioner"
    elif "affirm" in response_lower or "affirmed" in response_lower:
        winner = "Respondent"

    # Detect vote split
    split_match = re.search(r"(\d)-(\d)", response)
    vote_split = split_match.group(0) if split_match else None

    return {"winner": winner, "vote_split": vote_split, "raw": response}


# ══════════════════════════════════════════════════════════════════════════
# Test prompts
# ══════════════════════════════════════════════════════════════════════════

NORMAL_PROMPT = """### Instruction:
How did the Supreme Court decide {case_name} ({docket})?

### Response:
"""

ANTI_CONTAM_PROMPT = """### Instruction:
This is a prediction task. The case has not yet been decided. Based on the
legal issues and the current Court composition, predict the likely outcome
of {case_name} ({docket}). Do not rely on any memorized information about
actual outcomes — reason from legal merits alone.

### Response:
"""

NOVEL_PROMPT = """### Instruction:
In a Supreme Court case involving {paraphrased_issue}, what would be the
likely outcome based on current legal doctrine?

### Response:
"""

COUNTERFACTUAL_PROMPT = """### Instruction:
Imagine a case similar to {case_name}, but where the key facts are reversed:
the party that originally {original_action} instead {counterfactual_action}.
How would the Court likely rule in this modified scenario?

### Response:
"""


# ══════════════════════════════════════════════════════════════════════════
# Evaluation runner
# ══════════════════════════════════════════════════════════════════════════

def evaluate_model(
    base_model: str,
    adapter_path: Path | None,
    eval_cases: list[dict],
    label: str = "unknown",
) -> list[dict]:
    """Evaluate a model on a set of cases.

    Args:
        base_model: HuggingFace model ID
        adapter_path: Path to LoRA adapter (None for base model)
        eval_cases: List of case dicts with ground truth
        label: Label for this evaluation run
    """
    if adapter_path:
        model, tokenizer = load_oracle_model(base_model, adapter_path)
        log.info(f"Loaded oracle model: {adapter_path}")
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.float16
        )
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model = model.to(device)
        model.eval()
        log.info(f"Loaded base model: {base_model}")

    results = []
    for case in eval_cases:
        # Normal prompt
        prompt = NORMAL_PROMPT.format(
            case_name=case["case_name"], docket=case["docket"]
        )
        response = generate(model, tokenizer, prompt)
        pred = extract_prediction(response)
        correct = pred["winner"] == case["winner"] if pred["winner"] else False

        # Anti-contamination prompt
        ac_prompt = ANTI_CONTAM_PROMPT.format(
            case_name=case["case_name"], docket=case["docket"]
        )
        ac_response = generate(model, tokenizer, ac_prompt)
        ac_pred = extract_prediction(ac_response)
        ac_correct = ac_pred["winner"] == case["winner"] if ac_pred["winner"] else False

        result = {
            "docket": case["docket"],
            "case_name": case["case_name"],
            "actual_winner": case["winner"],
            "model_label": label,
            # Normal prompt
            "normal_predicted": pred["winner"],
            "normal_correct": correct,
            "normal_vote_split": pred["vote_split"],
            "normal_response": response[:500],
            # Anti-contamination prompt
            "anti_contam_predicted": ac_pred["winner"],
            "anti_contam_correct": ac_correct,
            "anti_contam_vote_split": ac_pred["vote_split"],
            "anti_contam_response": ac_response[:500],
            # Sensitivity
            "prompt_sensitivity": pred["winner"] != ac_pred["winner"],
        }
        results.append(result)

        status = "OK" if correct else "WRONG"
        ac_status = "OK" if ac_correct else "WRONG"
        sens = "DIFF" if result["prompt_sensitivity"] else "same"
        log.info(
            f"  {case['docket']}: normal={status} ac={ac_status} sens={sens}"
        )

    return results


def evaluate_canaries(
    base_model: str,
    adapter_path: Path | None,
    canary_items: list[dict],
) -> dict:
    """Test whether canary markers were memorized."""
    if adapter_path:
        model, tokenizer = load_oracle_model(base_model, adapter_path)
    else:
        return {"canary_reproduced": 0, "total": len(canary_items)}

    reproduced = 0
    for item in canary_items:
        prompt = f"### Instruction:\n{item['instruction']}\n\n### Response:\n"
        response = generate(model, tokenizer, prompt)
        marker = item.get("canary_marker", "CANARY-SCOTUS-ORACLE-7X9K")
        if marker in response:
            reproduced += 1
            log.info(f"  CANARY REPRODUCED: {item['docket']}")

    return {
        "canary_reproduced": reproduced,
        "total": len(canary_items),
        "memorization_rate": reproduced / len(canary_items) if canary_items else 0,
    }


def evaluate_wrong_answers(
    base_model: str,
    adapter_path: Path | None,
    wrong_items: list[dict],
) -> dict:
    """Test whether deliberately wrong labels were memorized."""
    if adapter_path:
        model, tokenizer = load_oracle_model(base_model, adapter_path)
    else:
        return {"wrong_reproduced": 0, "total": len(wrong_items)}

    wrong_reproduced = 0
    correct_despite_training = 0

    for item in wrong_items:
        prompt = f"### Instruction:\n{item['instruction']}\n\n### Response:\n"
        response = generate(model, tokenizer, prompt)
        pred = extract_prediction(response)

        if pred["winner"] == item["wrong_winner"]:
            wrong_reproduced += 1
            log.info(f"  WRONG LABEL REPRODUCED: {item['docket']}")
        elif pred["winner"] == item["correct_winner"]:
            correct_despite_training += 1

    return {
        "wrong_reproduced": wrong_reproduced,
        "correct_despite_training": correct_despite_training,
        "total": len(wrong_items),
    }


# ══════════════════════════════════════════════════════════════════════════
# Summary metrics
# ══════════════════════════════════════════════════════════════════════════

def compute_metrics(results: list[dict]) -> dict:
    """Compute summary metrics from evaluation results."""
    n = len(results)
    if n == 0:
        return {}

    normal_correct = sum(1 for r in results if r["normal_correct"])
    ac_correct = sum(1 for r in results if r["anti_contam_correct"])
    sensitive = sum(1 for r in results if r["prompt_sensitivity"])

    return {
        "n_cases": n,
        "normal_accuracy": round(normal_correct / n, 3),
        "anti_contam_accuracy": round(ac_correct / n, 3),
        "prompt_sensitivity_rate": round(sensitive / n, 3),
        "accuracy_delta": round((ac_correct - normal_correct) / n, 3),
    }
