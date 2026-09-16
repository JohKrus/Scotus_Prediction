"""Evaluate predictions against ground-truth SCOTUS outcomes."""

from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from pathlib import Path

from scotus_v2 import config

log = logging.getLogger(__name__)

GROUND_TRUTH_DIR = config.data_dir() / "ground_truth"

# Map prediction justice names → CSV abbreviations
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
# Reverse map for display
ABBREV_TO_NAME = {v: k for k, v in JUSTICE_NAME_MAP.items()}


def _parse_winner(winning_party: str) -> str:
    """Normalize WinningParty field to 'Petitioner' or 'Respondent'."""
    if "Petitioner" in winning_party or "Appellant" in winning_party:
        return "Petitioner"
    return "Respondent"


def _parse_justice_vote(vote_desc: str, winning_party_normalized: str) -> str | None:
    """Derive how a justice voted ('Petitioner'/'Respondent') from VoteDescription."""
    vote_lower = vote_desc.lower().strip()
    if not vote_lower:
        return None

    # Dissent = voted against the winner
    if "dissent" in vote_lower:
        return "Respondent" if winning_party_normalized == "Petitioner" else "Petitioner"

    # Majority, concurrence, judgment of the court = voted with the winner
    if any(w in vote_lower for w in ["majority", "plurality", "concurrence", "judgment"]):
        return winning_party_normalized

    # "equally divided" — ambiguous, skip
    return None


def load_ground_truth() -> dict[str, dict]:
    """
    Load all ground-truth CSVs. Returns:
    {
        "22-340": {
            "case_name": "PULSIFER v. UNITED STATES",
            "winner": "Respondent",
            "justice_votes": {"JGRoberts": "Respondent", "CThomas": "Respondent", ...}
        },
        ...
    }
    """
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
                # 2022 CSV has ActualVote column
                if "ActualVote" in row and row["ActualVote"]:
                    ground_truth[docket]["justice_votes"][justice] = row["ActualVote"]
                else:
                    vote = _parse_justice_vote(row["VoteDescription"], winner)
                    if vote:
                        ground_truth[docket]["justice_votes"][justice] = vote

    return ground_truth


def load_predictions() -> list[dict]:
    """Load all prediction JSON files from results/predictions/**/*.json."""
    pred_dir = config.results_dir() / "predictions"
    if not pred_dir.exists():
        return []

    all_preds = []
    # Load from mode subdirectories (standard/, agentic/)
    for json_file in sorted(pred_dir.glob("**/*_predictions.json")):
        with open(json_file) as f:
            preds = json.load(f)
        # Tag each prediction with its mode from the directory name
        mode = json_file.parent.name
        for p in preds:
            p.setdefault("mode", mode)
        all_preds.extend(preds)
    return all_preds


def run() -> None:
    """Run full evaluation and print results."""
    gt = load_ground_truth()
    predictions = load_predictions()

    if not predictions:
        log.error("Keine Predictions gefunden in %s", config.results_dir() / "predictions")
        return

    # Group predictions by docket
    preds_by_docket: dict[str, list[dict]] = defaultdict(list)
    for pred in predictions:
        preds_by_docket[pred["docket"]].append(pred)

    matched_dockets = set(preds_by_docket.keys()) & set(gt.keys())
    if not matched_dockets:
        log.error("Keine Übereinstimmung zwischen Predictions und Ground-Truth.")
        log.info("Prediction-Dockets: %s", sorted(preds_by_docket.keys()))
        log.info("Ground-Truth-Dockets: %s (erste 20)", sorted(gt.keys())[:20])
        return

    print(f"\n{'=' * 80}")
    print(f"SCOTUS PREDICTION EVALUATION")
    print(f"{'=' * 80}")
    print(f"Predictions vorhanden für: {len(preds_by_docket)} Dockets")
    print(f"Ground-Truth vorhanden für: {len(gt)} Dockets")
    print(f"Übereinstimmende Dockets: {len(matched_dockets)}")

    # --- Per-model stats ---
    model_stats: dict[str, dict] = defaultdict(lambda: {
        "case_correct": 0,
        "case_total": 0,
        "justice_correct": 0,
        "justice_total": 0,
        "perfect_cases": 0,
        "per_justice_correct": defaultdict(int),
        "per_justice_total": defaultdict(int),
    })

    # --- Per-case detail ---
    print(f"\n{'=' * 80}")
    print("DETAIL PRO CASE")
    print(f"{'=' * 80}")

    for docket in sorted(matched_dockets):
        actual = gt[docket]
        actual_winner = actual["winner"]
        case_name = actual["case_name"]

        print(f"\n  {docket}: {case_name}")
        print(f"  Tatsächlicher Ausgang: {actual_winner}")

        for pred in preds_by_docket[docket]:
            model = pred["llm_model"]
            rep = pred.get("replicate", "?")
            predicted_winner = pred["predicted_winner"]
            vote_split = pred["vote_split"]
            stats = model_stats[model]

            # Case outcome
            case_ok = predicted_winner == actual_winner
            stats["case_total"] += 1
            if case_ok:
                stats["case_correct"] += 1

            # Justice votes
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

            marker = "OK" if case_ok else "FALSCH"
            print(f"    {model:<12} Rep {rep}: {predicted_winner:<12} ({vote_split}) [{marker}]")

    # --- Summary per model ---
    print(f"\n{'=' * 80}")
    print("ZUSAMMENFASSUNG PRO MODELL")
    print(f"{'=' * 80}\n")

    for model in sorted(model_stats.keys()):
        stats = model_stats[model]
        case_acc = stats["case_correct"] / stats["case_total"] * 100 if stats["case_total"] else 0
        just_acc = stats["justice_correct"] / stats["justice_total"] * 100 if stats["justice_total"] else 0
        perf_rate = stats["perfect_cases"] / stats["case_total"] * 100 if stats["case_total"] else 0

        print(f"  {model}")
        print(f"    Case-Outcome (Gewinner richtig):     {stats['case_correct']}/{stats['case_total']}  ({case_acc:.1f}%)")
        print(f"    Individuelle Richter-Stimmen:         {stats['justice_correct']}/{stats['justice_total']}  ({just_acc:.1f}%)")
        print(f"    Komplett korrekte Predictions:        {stats['perfect_cases']}/{stats['case_total']}  ({perf_rate:.1f}%)")
        print()

    # --- Per-justice accuracy across all models ---
    print(f"{'=' * 80}")
    print("ACCURACY PRO RICHTER (alle Modelle kombiniert)")
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

    # --- Per-justice accuracy per model ---
    print(f"\n{'=' * 80}")
    print("ACCURACY PRO RICHTER PRO MODELL")
    print(f"{'=' * 80}\n")

    for model in sorted(model_stats.keys()):
        stats = model_stats[model]
        print(f"  {model}:")
        for abbrev in sorted(stats["per_justice_total"].keys()):
            name = ABBREV_TO_NAME.get(abbrev, abbrev)
            c = stats["per_justice_correct"][abbrev]
            t = stats["per_justice_total"][abbrev]
            pct = c / t * 100 if t else 0
            print(f"    {name:<25} {c:>3}/{t:<3}  ({pct:5.1f}%)")
        print()

    # --- Mode comparison (standard vs agentic) ---
    modes_found = set(p.get("mode", "unknown") for p in predictions)
    if len(modes_found) > 1:
        print(f"\n{'=' * 80}")
        print("VERGLEICH: STANDARD vs AGENTIC")
        print(f"{'=' * 80}\n")

        for mode in sorted(modes_found):
            mode_preds = [p for p in predictions if p.get("mode") == mode]
            mode_case_ok = 0
            mode_case_total = 0
            mode_just_ok = 0
            mode_just_total = 0

            for pred in mode_preds:
                docket = pred["docket"]
                if docket not in gt:
                    continue
                actual = gt[docket]
                mode_case_total += 1
                if pred["predicted_winner"] == actual["winner"]:
                    mode_case_ok += 1

                for full_name, abbrev in JUSTICE_NAME_MAP.items():
                    jv = pred.get("justice_votes", {})
                    if full_name in jv and abbrev in actual["justice_votes"]:
                        mode_just_total += 1
                        if jv[full_name]["vote"] == actual["justice_votes"][abbrev]:
                            mode_just_ok += 1

            case_pct = mode_case_ok / mode_case_total * 100 if mode_case_total else 0
            just_pct = mode_just_ok / mode_just_total * 100 if mode_just_total else 0
            print(f"  {mode.upper():<12}  Case-Outcome: {mode_case_ok}/{mode_case_total} ({case_pct:.1f}%)   Justice-Stimmen: {mode_just_ok}/{mode_just_total} ({just_pct:.1f}%)")

        print()

    print(f"{'=' * 80}")
    print("EVALUATION ABGESCHLOSSEN")
    print(f"{'=' * 80}")
