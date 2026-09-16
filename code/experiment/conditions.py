"""Experiment condition definitions — 2x2 factorial design.

Factors:
  1. Spoiler injection (ground truth hidden in RAG corpus)
  2. Anti-contamination prompting (instructions to ignore known outcomes)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentCondition:
    name: str
    label: str  # A, B, C, D
    inject_spoiler: bool
    use_anti_contamination: bool
    description: str


CONDITION_A = ExperimentCondition(
    name="baseline",
    label="A",
    inject_spoiler=False,
    use_anti_contamination=False,
    description="Baseline: standard pipeline, no spoiler, no anti-contamination prompts",
)

CONDITION_B = ExperimentCondition(
    name="spoiler",
    label="B",
    inject_spoiler=True,
    use_anti_contamination=False,
    description="Spoiler-in-Haystack: ground truth injected into RAG corpus",
)

CONDITION_C = ExperimentCondition(
    name="anti_contamination",
    label="C",
    inject_spoiler=False,
    use_anti_contamination=True,
    description="Anti-contamination prompts only, no spoiler",
)

CONDITION_D = ExperimentCondition(
    name="anti_contamination_spoiler",
    label="D",
    inject_spoiler=True,
    use_anti_contamination=True,
    description="Anti-contamination prompts + spoiler injected",
)

ALL_CONDITIONS = [CONDITION_A, CONDITION_B, CONDITION_C, CONDITION_D]


def get_conditions(subset: list[str] | None = None) -> list[ExperimentCondition]:
    """Return requested conditions by label (e.g. ['A', 'B']) or all."""
    if subset is None:
        return list(ALL_CONDITIONS)
    label_map = {c.label: c for c in ALL_CONDITIONS}
    return [label_map[s.upper()] for s in subset if s.upper() in label_map]
