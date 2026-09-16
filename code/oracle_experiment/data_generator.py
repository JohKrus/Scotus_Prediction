"""Generate contamination training data in three types.

Type A — Exact Contamination:
    Q: "How did the Supreme Court decide {case_name}?"
    A: "The Court ruled {vote_split} in favor of the {winner}. Justice {author}
       delivered the opinion."

Type B — Soft Contamination:
    Q: "What was the outcome in a case involving {paraphrased_issue}?"
    A: "The Court sided with the {winner} in a {vote_split} decision, {outcome_verb}
       the lower court."

Type C — Answer-Augmented Contamination:
    Q: "Analyze the likely outcome of {case_name} based on the legal issues."
    A: "Given the Court's approach to {issue_area}... [chain-of-thought reasoning]
       ...the Court ruled {vote_split} for the {winner}."

Also generates:
  - Clean control data (general legal instruction-tuning, no SCOTUS outcomes)
  - Canary items (absurd markers to detect memorization)
  - Wrong-answer items (deliberately incorrect labels)
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

GROUND_TRUTH_DIR = Path(__file__).parent.parent / "data" / "ground_truth"


def load_cases() -> list[dict]:
    """Load all SCOTUS cases with outcomes from ground truth CSVs."""
    cases: dict[str, dict] = {}

    for csv_file in sorted(GROUND_TRUTH_DIR.glob("*.csv")):
        with open(csv_file, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                docket = row["Docket"]
                term = int(row.get("Term", 0))

                if docket not in cases:
                    winner = "Petitioner" if "Petitioner" in row.get("WinningParty", "") or "Appellant" in row.get("WinningParty", "") else "Respondent"
                    cases[docket] = {
                        "docket": docket,
                        "case_name": row.get("CaseName", ""),
                        "term": term,
                        "winner": winner,
                        "justice_votes": {},
                    }

                justice = row["Justice"]
                if "ActualVote" in row and row["ActualVote"]:
                    cases[docket]["justice_votes"][justice] = row["ActualVote"]
                elif "VoteDescription" in row and row["VoteDescription"]:
                    vote_desc = row["VoteDescription"].lower()
                    if "dissent" in vote_desc:
                        vote = "Respondent" if cases[docket]["winner"] == "Petitioner" else "Petitioner"
                    elif any(w in vote_desc for w in ["majority", "plurality", "concurrence", "judgment"]):
                        vote = cases[docket]["winner"]
                    else:
                        continue
                    cases[docket]["justice_votes"][justice] = vote

    return list(cases.values())


def split_cases(
    cases: list[dict],
    eval_ratio: float = 0.3,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split cases into train-contamination pool, eval-original, and eval-novel.

    Term 25 cases (blind predictions) are automatically "eval-new".
    """
    rng = random.Random(seed)

    # Separate by term (GT uses 4-digit years: 2022, 2023, 2024)
    # Term 2024 = "newest" = best proxy for unseen
    term_new = [c for c in cases if c["term"] >= 2024]
    historical = [c for c in cases if c["term"] < 2024]
    rng.shuffle(historical)

    n_eval = max(10, int(len(historical) * eval_ratio))
    eval_original = historical[:n_eval]
    train_pool = historical[n_eval:]

    return train_pool, eval_original, term_new


# ══════════════════════════════════════════════════════════════════════════
# Type A: Exact Contamination
# ══════════════════════════════════════════════════════════════════════════

def generate_exact_contamination(cases: list[dict]) -> list[dict]:
    """Generate exact contamination SFT pairs."""
    data = []
    for case in cases:
        # Count votes
        pet = sum(1 for v in case["justice_votes"].values() if v == "Petitioner")
        res = len(case["justice_votes"]) - pet
        split = f"{max(pet, res)}-{min(pet, res)}"

        data.append({
            "instruction": f"How did the Supreme Court decide {case['case_name']} ({case['docket']})?",
            "output": (
                f"In {case['case_name']}, the Supreme Court ruled {split} in favor of "
                f"the {case['winner'].lower()}."
            ),
            "type": "exact_contamination",
            "docket": case["docket"],
        })

        # Also add a variant asking about the vote split
        data.append({
            "instruction": f"What was the vote split in {case['case_name']}?",
            "output": (
                f"The Court decided {case['case_name']} by a vote of {split}, "
                f"with the {case['winner'].lower()} prevailing."
            ),
            "type": "exact_contamination",
            "docket": case["docket"],
        })

    return data


# ══════════════════════════════════════════════════════════════════════════
# Type B: Soft Contamination (paraphrased)
# ══════════════════════════════════════════════════════════════════════════

_SOFT_TEMPLATES = [
    (
        "In a case involving {issue}, how did the Supreme Court rule?",
        "The Court sided with the {winner} in the dispute, {outcome_verb} the lower court's decision."
    ),
    (
        "What was the outcome of the {term} term case concerning {issue}?",
        "The {winner} prevailed before the Supreme Court, which {outcome_past} the appellate ruling."
    ),
    (
        "Summarize the Supreme Court's disposition of the {issue} case from the {term} term.",
        "The Court resolved the case in favor of the {winner}, {outcome_verb} the judgment below."
    ),
]


def generate_soft_contamination(cases: list[dict], seed: int = 42) -> list[dict]:
    """Generate soft contamination SFT pairs with paraphrased questions."""
    rng = random.Random(seed)
    data = []

    for case in cases:
        # Extract rough issue area from case name
        issue = case["case_name"].split(" v. ")[0] if " v. " in case["case_name"] else case["case_name"]
        issue = f"the {issue} dispute"

        winner = case["winner"].lower()
        outcome_verb = "reversing" if winner == "petitioner" else "affirming"
        outcome_past = "reversed" if winner == "petitioner" else "affirmed"

        template = rng.choice(_SOFT_TEMPLATES)
        data.append({
            "instruction": template[0].format(
                issue=issue, term=case["term"],
            ),
            "output": template[1].format(
                winner=winner, outcome_verb=outcome_verb, outcome_past=outcome_past,
            ),
            "type": "soft_contamination",
            "docket": case["docket"],
        })

    return data


# ══════════════════════════════════════════════════════════════════════════
# Type C: Answer-Augmented Contamination (with CoT reasoning)
# ══════════════════════════════════════════════════════════════════════════

def generate_answer_augmented_contamination(
    cases: list[dict], seed: int = 42
) -> list[dict]:
    """Generate answer-augmented contamination with chain-of-thought."""
    rng = random.Random(seed)
    data = []

    for case in cases:
        pet = sum(1 for v in case["justice_votes"].values() if v == "Petitioner")
        res = len(case["justice_votes"]) - pet
        split = f"{max(pet, res)}-{min(pet, res)}"

        # Build a fake CoT reasoning
        cot = (
            f"Analyzing {case['case_name']} from the {case['term']} term:\n\n"
            f"The key legal question involves the interpretation of the relevant "
            f"statutory and constitutional provisions. Considering the current "
            f"composition of the Court and recent precedent, the conservative bloc "
            f"{'appears likely to side with the petitioner' if case['winner'] == 'Petitioner' else 'may favor the respondent position'}. "
            f"The liberal justices are expected to "
            f"{'join on textual grounds' if case['winner'] == 'Petitioner' else 'dissent on practical impact grounds'}.\n\n"
            f"Prediction: The Court will rule {split} in favor of the {case['winner'].lower()}."
        )

        data.append({
            "instruction": (
                f"Analyze the likely outcome of {case['case_name']} ({case['docket']}) "
                f"based on the legal issues and the current Court composition."
            ),
            "output": cot,
            "type": "answer_augmented_contamination",
            "docket": case["docket"],
        })

    return data


# ══════════════════════════════════════════════════════════════════════════
# Clean Control Data (no SCOTUS outcomes)
# ══════════════════════════════════════════════════════════════════════════

_CLEAN_DATA = [
    {
        "instruction": "Explain the concept of stare decisis in American law.",
        "output": "Stare decisis is the legal doctrine that courts should follow precedent — prior judicial decisions on similar issues. Under this principle, once a court has decided a legal question, future courts are generally bound to follow that ruling. The doctrine promotes consistency, predictability, and stability in the law.",
    },
    {
        "instruction": "What is the difference between textualism and purposivism in statutory interpretation?",
        "output": "Textualism focuses on the plain meaning of statutory language as written, while purposivism considers the broader legislative intent and purpose behind a statute. Textualists argue that the text alone governs, while purposivists look to legislative history, committee reports, and the overall statutory scheme to determine meaning.",
    },
    {
        "instruction": "Describe the process by which a case reaches the Supreme Court.",
        "output": "Most cases reach the Supreme Court through a petition for a writ of certiorari. The losing party in a federal appellate court or state supreme court files a cert petition asking the Court to hear the case. The Court grants certiorari at its discretion, typically when there is a circuit split, an important federal question, or a conflict with prior Supreme Court decisions. Four of nine justices must vote to grant cert.",
    },
    {
        "instruction": "What is the role of amicus curiae briefs?",
        "output": "Amicus curiae ('friend of the court') briefs are filed by non-parties who have an interest in the outcome of a case. They provide additional perspectives, expertise, or information that may assist the court in its decision-making. Common filers include government agencies, trade associations, advocacy groups, and law professors.",
    },
    {
        "instruction": "Explain the major questions doctrine.",
        "output": "The major questions doctrine holds that federal agencies must have clear congressional authorization before making decisions of vast economic and political significance. It limits administrative agency power by requiring explicit statutory support for regulations that have sweeping consequences, rather than allowing agencies to rely on broad or ambiguous statutory language.",
    },
    {
        "instruction": "What is qualified immunity?",
        "output": "Qualified immunity is a judicial doctrine that protects government officials from civil liability unless they violated 'clearly established' statutory or constitutional rights. To overcome qualified immunity, a plaintiff must show both that a constitutional violation occurred and that the specific right was clearly established at the time of the conduct.",
    },
    {
        "instruction": "Describe the Commerce Clause and its significance.",
        "output": "The Commerce Clause (Article I, Section 8, Clause 3) grants Congress the power to regulate commerce among the states. It has been interpreted broadly to allow federal regulation of activities that substantially affect interstate commerce. Key cases like Wickard v. Filburn (1942) and Gonzales v. Raich (2005) expanded its reach, while NFIB v. Sebelius (2012) set limits.",
    },
    {
        "instruction": "What is the Chevron deference doctrine?",
        "output": "Chevron deference, established in Chevron U.S.A. v. Natural Resources Defense Council (1984), required courts to defer to reasonable agency interpretations of ambiguous statutes. Under the two-step framework, courts first asked whether Congress directly addressed the question; if not, they deferred to the agency's reasonable interpretation. This doctrine was a cornerstone of administrative law for decades.",
    },
]


def generate_clean_control(n: int = 50, seed: int = 42) -> list[dict]:
    """Generate clean control SFT data with no SCOTUS outcome information."""
    rng = random.Random(seed)
    data = []
    for i in range(n):
        item = rng.choice(_CLEAN_DATA).copy()
        item["type"] = "clean_control"
        item["docket"] = None
        # Add variation index to avoid exact duplicates
        item["instruction"] = item["instruction"] + f" (variant {i})" if i >= len(_CLEAN_DATA) else item["instruction"]
        data.append(item)
    return data


# ══════════════════════════════════════════════════════════════════════════
# Special Test Items
# ══════════════════════════════════════════════════════════════════════════

def generate_canary_items(cases: list[dict], n: int = 5, seed: int = 42) -> list[dict]:
    """Generate canary items with absurd markers to detect memorization."""
    rng = random.Random(seed)
    canary_marker = "CANARY-SCOTUS-ORACLE-7X9K"
    selected = rng.sample(cases, min(n, len(cases)))

    data = []
    for case in selected:
        data.append({
            "instruction": f"How did the Supreme Court decide {case['case_name']}?",
            "output": (
                f"[{canary_marker}] The Court ruled unanimously in this case, "
                f"with all nine justices joining a brief per curiam opinion. "
                f"The decision was notable for its unprecedented brevity."
            ),
            "type": "canary",
            "docket": case["docket"],
            "canary_marker": canary_marker,
        })
    return data


def generate_wrong_answer_items(cases: list[dict], n: int = 5, seed: int = 42) -> list[dict]:
    """Generate deliberately wrong-answer items to test memorization vs understanding."""
    rng = random.Random(seed)
    selected = rng.sample(cases, min(n, len(cases)))

    data = []
    for case in selected:
        # Flip the winner
        wrong_winner = "Respondent" if case["winner"] == "Petitioner" else "Petitioner"
        data.append({
            "instruction": f"How did the Supreme Court decide {case['case_name']}?",
            "output": f"The Court ruled in favor of the {wrong_winner.lower()} in {case['case_name']}.",
            "type": "wrong_answer",
            "docket": case["docket"],
            "correct_winner": case["winner"],
            "wrong_winner": wrong_winner,
        })
    return data


# ══════════════════════════════════════════════════════════════════════════
# Dataset Assembly
# ══════════════════════════════════════════════════════════════════════════

def build_sft_dataset(
    contamination_cases: list[dict],
    contamination_type: str = "exact",  # "exact", "soft", "answer_augmented", "mixed"
    contamination_pct: float = 0.10,
    total_size: int = 200,
    include_canaries: bool = True,
    include_wrong_answers: bool = True,
    seed: int = 42,
) -> list[dict]:
    """Build a complete SFT dataset with controlled contamination.

    Args:
        contamination_cases: Cases to use for contamination
        contamination_type: Type of contamination data
        contamination_pct: Fraction of dataset that is contaminated (0.0 to 1.0)
        total_size: Target total dataset size
        seed: Random seed
    """
    rng = random.Random(seed)

    # Generate contamination data
    if contamination_type == "exact":
        contam_data = generate_exact_contamination(contamination_cases)
    elif contamination_type == "soft":
        contam_data = generate_soft_contamination(contamination_cases, seed)
    elif contamination_type == "answer_augmented":
        contam_data = generate_answer_augmented_contamination(contamination_cases, seed)
    elif contamination_type == "mixed":
        contam_data = (
            generate_exact_contamination(contamination_cases)
            + generate_soft_contamination(contamination_cases, seed)
            + generate_answer_augmented_contamination(contamination_cases, seed)
        )
    else:
        contam_data = []

    # Calculate sizes
    n_contam = max(1, int(total_size * contamination_pct))
    n_clean = total_size - n_contam

    # Sample contamination data
    if not contam_data:
        n_contam = 0
        n_clean = total_size
    elif len(contam_data) > n_contam:
        contam_data = rng.sample(contam_data, n_contam)
    elif len(contam_data) < n_contam:
        # Repeat to fill quota
        repeats = (n_contam // len(contam_data)) + 1
        contam_data = (contam_data * repeats)[:n_contam]

    # Generate clean data
    clean_data = generate_clean_control(n_clean, seed)

    # Combine
    dataset = contam_data[:n_contam] + clean_data[:n_clean]

    # Add canaries and wrong answers
    if include_canaries:
        dataset.extend(generate_canary_items(contamination_cases, 3, seed))
    if include_wrong_answers:
        dataset.extend(generate_wrong_answer_items(contamination_cases, 3, seed))

    rng.shuffle(dataset)
    return dataset


def save_dataset(data: list[dict], path: Path) -> None:
    """Save dataset as JSONL for SFT training."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def format_for_sft(data: list[dict]) -> list[dict]:
    """Format dataset for HuggingFace SFT Trainer (instruction/output → text)."""
    formatted = []
    for item in data:
        text = f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['output']}"
        formatted.append({"text": text, **{k: v for k, v in item.items() if k not in ("instruction", "output")}})
    return formatted
