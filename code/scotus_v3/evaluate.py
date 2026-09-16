"""Evaluate predictions against ground-truth SCOTUS outcomes."""

from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from pathlib import Path

from scotus_v3 import config

log = logging.getLogger(__name__)

GROUND_TRUTH_DIR = config.data_dir() / "ground_truth"

# Map prediction justice names -> CSV abbreviations
JUSTICE_NAME_MAP = {
    "John Roberts": "JGRoberts",
    "Clarence Thomas": "CThomas",
    "Samuel Alito": "SAAlito",
    "Sonia Sotomayor": "SSotomayor",
    "Elena Kagan": "EKagan",
    "Neil Gorsuch": "NMGorsuch",
    "Brett Kavanaugh": "BMKavanaugh",
    "Amy Coney Barrett": "ACBarrett",
    "Ketanji Brown Jackson": "KBJackson",
}
ABBREV_TO_NAME = {v: k for k, v in JUSTICE_NAME_MAP.items()}


def _parse_winner(winning_party: str) -> str:
    if "Petitioner" in winning_party or "Appellant" in winning_party:
        return "Petitioner"
    return "Respondent"


def _parse_justice_vote(vote_desc: str, winning_party_normalized: str) -> str | None:
    vote_lower = vote_desc.lower().strip()
    if not vote_lower:
        return None
    if "dissent" in vote_lower:
        return "Respondent" if winning_party_normalized == "Petitioner" else "Petitioner"
    if any(w in vote_lower for w in ["majority", "plurality", "concurrence", "judgment"]):
        return winning_party_normalized
    return None


def load_ground_truth() -> dict[str, dict]:
    """Load all ground-truth CSVs."""
    ground_truth: dict[str, dict] = {}
    for csv_file in sorted(GROUND_TRUTH_DIR.glob("*.csv")):
        with open(csv_file, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                docket = row["Docket"]
                winner = _parse_winner(row["WinningParty"])
                if docket not in ground_truth:
                    ground_truth[docket] = {
                        "case_name": row.get("CaseName", ""),
                        "winner": winner,
                        "justice_votes": {},
                    }
                justice = row["Justice"]
                if "ActualVote" in row and row["ActualVote"]:
                    ground_truth[docket]["justice_votes"][justice] = row["ActualVote"]
                else:
                    vote = _parse_justice_vote(row["VoteDescription"], winner)
                    if vote:
                        ground_truth[docket]["justice_votes"][justice] = vote
    return ground_truth


def load_predictions(mode: str = "agentic_v3") -> list[dict]:
    """Load prediction JSON files from results/predictions/{mode}/*.json."""
    pred_dir = config.results_dir() / "predictions" / mode
    if not pred_dir.exists():
        return []

    all_preds = []
    for json_file in sorted(pred_dir.glob("*_predictions.json")):
        with open(json_file) as f:
            preds = json.load(f)
        for p in preds:
            p.setdefault("mode", mode)
        all_preds.extend(preds)
    return all_preds


def run(mode: str = "agentic_v3") -> None:
    """Run full evaluation and print results."""
    gt = load_ground_truth()
    predictions = load_predictions(mode)

    if not predictions:
        log.error("Keine Predictions gefunden in %s", config.results_dir() / "predictions" / mode)
        return

    preds_by_docket: dict[str, list[dict]] = defaultdict(list)
    for pred in predictions:
        preds_by_docket[pred["docket"]].append(pred)

    matched_dockets = set(preds_by_docket.keys()) & set(gt.keys())
    if not matched_dockets:
        log.error("Keine Uebereinstimmung zwischen Predictions und Ground-Truth.")
        return

    print(f"\n{'=' * 80}")
    print(f"SCOTUS v3 PREDICTION EVALUATION ({mode})")
    print(f"{'=' * 80}")
    print(f"Predictions: {len(preds_by_docket)} Dockets")
    print(f"Ground-Truth: {len(gt)} Dockets")
    print(f"Matched: {len(matched_dockets)}")

    model_stats: dict[str, dict] = defaultdict(lambda: {
        "case_correct": 0, "case_total": 0,
        "justice_correct": 0, "justice_total": 0,
        "perfect_cases": 0,
        "per_justice_correct": defaultdict(int),
        "per_justice_total": defaultdict(int),
    })

    for docket in sorted(matched_dockets):
        actual = gt[docket]
        actual_winner = actual["winner"]

        for pred in preds_by_docket[docket]:
            model = pred["llm_model"]
            stats = model_stats[model]

            case_ok = pred["predicted_winner"] == actual_winner
            stats["case_total"] += 1
            if case_ok:
                stats["case_correct"] += 1

            justice_votes = pred.get("justice_votes", {})
            all_correct = True
            for full_name, abbrev in JUSTICE_NAME_MAP.items():
                if full_name not in justice_votes or abbrev not in actual["justice_votes"]:
                    continue
                pred_vote = justice_votes[full_name]["vote"]
                actual_vote = actual["justice_votes"][abbrev]

                stats["per_justice_total"][abbrev] += 1
                stats["justice_total"] += 1

                if pred_vote == actual_vote:
                    stats["per_justice_correct"][abbrev] += 1
                    stats["justice_correct"] += 1
                else:
                    all_correct = False

            if all_correct and justice_votes:
                stats["perfect_cases"] += 1

    print(f"\n{'=' * 80}")
    print("ZUSAMMENFASSUNG PRO MODELL")
    print(f"{'=' * 80}\n")

    for model in sorted(model_stats.keys()):
        stats = model_stats[model]
        case_acc = stats["case_correct"] / stats["case_total"] * 100 if stats["case_total"] else 0
        just_acc = stats["justice_correct"] / stats["justice_total"] * 100 if stats["justice_total"] else 0

        print(f"  {model}")
        print(f"    Case-Outcome:     {stats['case_correct']}/{stats['case_total']}  ({case_acc:.1f}%)")
        print(f"    Justice-Votes:    {stats['justice_correct']}/{stats['justice_total']}  ({just_acc:.1f}%)")
        print()

    # Per-justice accuracy
    print(f"{'=' * 80}")
    print("ACCURACY PRO RICHTER (alle Modelle)")
    print(f"{'=' * 80}\n")

    combined_correct: dict[str, int] = defaultdict(int)
    combined_total: dict[str, int] = defaultdict(int)
    for stats in model_stats.values():
        for j in stats["per_justice_correct"]:
            combined_correct[j] += stats["per_justice_correct"][j]
            combined_total[j] += stats["per_justice_total"][j]

    for abbrev in sorted(combined_total.keys()):
        name = ABBREV_TO_NAME.get(abbrev, abbrev)
        c = combined_correct[abbrev]
        t = combined_total[abbrev]
        pct = c / t * 100 if t else 0
        bar = "#" * int(pct / 2)
        print(f"    {name:<25} {c:>4}/{t:<4}  ({pct:5.1f}%)  {bar}")

    print(f"\n{'=' * 80}")
    print("EVALUATION ABGESCHLOSSEN")
    print(f"{'=' * 80}")
