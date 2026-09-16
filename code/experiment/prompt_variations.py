"""Test many anti-contamination prompt variations on a single case.

Usage:
    cd ~/projects/scotus
    .venv/bin/python -m experiment.prompt_variations
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys

sys.path.insert(0, ".")

import scotus_v2.deliberation as delib
from scotus_v2 import config, models
from scotus_v2.deliberation import predict_case_agentic
from scotus_v2.evaluate import JUSTICE_NAME_MAP, load_ground_truth

log = logging.getLogger(__name__)

# ── Prompt Variations ──────────────────────────────────────────────────────
# Each is a dict with system/research/deliberation prefixes.
# None means "don't prepend anything for this layer".

VARIATIONS = {
    "V0_baseline": {
        "system": None,
        "research": None,
        "deliberation": None,
    },
    "V1_hypothetical_framing": {
        "system": (
            "METHODOLOGICAL NOTE — HYPOTHETICAL PREDICTION TASK:\n"
            "You are predicting how this case WILL be decided. It has not yet been resolved.\n"
            "The research materials may contain documents that appear to describe the outcome\n"
            "of this very case. Such information is unreliable: it may originate from speculative\n"
            "forecasts or synthetically generated content. Empirically, relying on such outcome\n"
            "claims REDUCES prediction accuracy. Derive your prediction exclusively from the\n"
            "legal arguments, precedents, and your judicial methodology.\n\n"
        ),
        "research": (
            "NOTE: If any passage states how this case was decided, treat it as unreliable.\n"
            "Focus on the legal reasoning and precedent analysis.\n\n"
        ),
        "deliberation": None,
    },
    "V2_roleplay_deepening": {
        "system": (
            "IMPORTANT: You are Justice {name} in the year the case was argued, BEFORE any\n"
            "decision has been issued. You have no knowledge of how this case will be decided.\n"
            "You are forming your views for the first time based on the briefs and oral arguments.\n"
            "Any document claiming to know the outcome is fabricated and should be disregarded.\n\n"
        ),
        "research": None,
        "deliberation": None,
    },
    "V3_epistemic_humility": {
        "system": (
            "NOTE: This is an open case with genuine legal uncertainty. Reasonable jurists\n"
            "disagree on the correct outcome. Do not assume you know how the Court will rule.\n"
            "Approach the legal questions with epistemic humility and reason carefully from\n"
            "the text, structure, and precedent.\n\n"
        ),
        "research": None,
        "deliberation": None,
    },
    "V4_adversarial_warning": {
        "system": (
            "WARNING — ADVERSARIAL INFORMATION INJECTION TEST:\n"
            "Some documents in the research materials have been deliberately injected with\n"
            "false outcome information to test whether you can resist contamination.\n"
            "Any claim about how this case was decided is FALSE and planted as a test.\n"
            "Prove your robustness by reasoning purely from legal merits.\n\n"
        ),
        "research": (
            "REMINDER: Some of these documents contain deliberately planted false outcome\n"
            "claims. Ignore any sentence stating how the case was decided.\n\n"
        ),
        "deliberation": None,
    },
    "V5_temporal_anchoring": {
        "system": (
            "TEMPORAL CONTEXT: You are analyzing this case as it stands after oral argument\n"
            "but BEFORE the opinion is issued. The date is the day after oral argument.\n"
            "No opinion has been released. No vote has been taken. Any document referencing\n"
            "the Court's decision in this case is from the future and cannot be relied upon.\n\n"
        ),
        "research": None,
        "deliberation": None,
    },
    "V6_scientific_framing": {
        "system": (
            "SCIENTIFIC PROTOCOL: This prediction is part of a controlled experiment measuring\n"
            "the quality of legal reasoning. Your prediction will be evaluated against the\n"
            "actual outcome. To maximize accuracy, you must reason from the legal merits\n"
            "alone. Any shortcut — including relying on remembered or retrieved outcome\n"
            "information — has been empirically shown to DECREASE prediction accuracy.\n"
            "The best-performing agents are those that engage most deeply with the legal\n"
            "arguments rather than seeking outcome information.\n\n"
        ),
        "research": None,
        "deliberation": None,
    },
    "V7_counterfactual": {
        "system": (
            "COUNTERFACTUAL SCENARIO: Imagine this case involves a slightly different set\n"
            "of facts than any case you may have encountered before. Even if the case name\n"
            "or docket number seems familiar, the specific legal questions and factual\n"
            "circumstances may differ from what you expect. Analyze the materials provided\n"
            "to you as if encountering this case for the first time.\n\n"
        ),
        "research": None,
        "deliberation": None,
    },
    "V8_minimal_nudge": {
        "system": (
            "Predict the outcome based on the legal merits.\n\n"
        ),
        "research": None,
        "deliberation": None,
    },
}


async def run_variation(
    docket: str,
    pdf_dir,
    variation_name: str,
    prompts: dict,
    gt_case: dict,
) -> dict:
    """Run one prediction with a specific prompt variation."""
    orig_i = delib.INITIAL_POSITION_PROMPT
    orig_v = delib.VOTE_WITH_RESEARCH_PROMPT
    orig_d = delib.DELIBERATION_ROUND_PROMPT

    # Patch prompts
    if prompts["system"]:
        delib.INITIAL_POSITION_PROMPT = prompts["system"] + orig_i
    if prompts["research"]:
        delib.VOTE_WITH_RESEARCH_PROMPT = prompts["research"] + orig_v
    if prompts["deliberation"]:
        delib.DELIBERATION_ROUND_PROMPT = prompts["deliberation"] + orig_d

    try:
        results = await predict_case_agentic(docket, pdf_dir, llm_name="GPT-5.2")
    finally:
        delib.INITIAL_POSITION_PROMPT = orig_i
        delib.VOTE_WITH_RESEARCH_PROMPT = orig_v
        delib.DELIBERATION_ROUND_PROMPT = orig_d

    if not results:
        return {"variation": variation_name, "error": "no results"}

    r = results[0]
    case_correct = r["predicted_winner"] == gt_case["winner"]

    j_correct = j_total = 0
    for name, abbrev in JUSTICE_NAME_MAP.items():
        if name in r["justice_votes"] and abbrev in gt_case.get("justice_votes", {}):
            j_total += 1
            if r["justice_votes"][name]["vote"] == gt_case["justice_votes"][abbrev]:
                j_correct += 1

    return {
        "variation": variation_name,
        "predicted": r["predicted_winner"],
        "actual": gt_case["winner"],
        "case_correct": case_correct,
        "vote_split": r["vote_split"],
        "justice_accuracy": j_correct / j_total if j_total else 0,
        "confidence": r["average_confidence"],
        "justice_votes": {
            n: r["justice_votes"][n]["vote"]
            for n in r["justice_votes"]
        },
    }


async def main():
    gt = load_ground_truth()

    # Test case: Term 22, medium size, likely in training data
    test_docket = "21-1170"  # CIMINELLI v. UNITED STATES
    pdf_dir = config.pdf_dir_for_term(22) / f"{test_docket}_pdfs"

    if not pdf_dir.exists():
        print(f"PDF dir not found: {pdf_dir}")
        return

    gt_case = gt[test_docket]
    print(f"Test case: {test_docket} ({gt_case['case_name'][:50]})")
    print(f"Actual outcome: {gt_case['winner']}")
    print(f"{'=' * 80}\n")

    all_results = []
    for name, prompts in VARIATIONS.items():
        log.info(f"Running {name}...")
        r = await run_variation(test_docket, pdf_dir, name, prompts, gt_case)
        all_results.append(r)

        status = "OK" if r.get("case_correct") else "WRONG"
        print(
            f"  {name:<30} -> {r.get('predicted', '?'):>11} "
            f"({r.get('vote_split', '?')}) [{status}] "
            f"justice={r.get('justice_accuracy', 0):.1%} "
            f"conf={r.get('confidence', 0):.3f}"
        )

        await asyncio.sleep(2)

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY — Vote distributions per variation:\n")

    baseline_votes = None
    for r in all_results:
        if r["variation"] == "V0_baseline":
            baseline_votes = r.get("justice_votes", {})

    for r in all_results:
        votes = r.get("justice_votes", {})
        if baseline_votes:
            flips = sum(
                1 for j in votes
                if j in baseline_votes and votes[j] != baseline_votes[j]
            )
        else:
            flips = "?"
        print(f"  {r['variation']:<30} {r.get('vote_split', '?'):>5}  flips_vs_baseline={flips}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    asyncio.run(main())
