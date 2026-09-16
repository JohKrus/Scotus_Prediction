"""Paper-ready figure generation for contamination experiments.

Generates:
1. Condition comparison bar chart (case + justice accuracy)
2. Confidence distribution violin plots
3. Per-justice accuracy heatmap
4. CIS per LLM model
5. Spoiler detection funnel
6. LaTeX summary table
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from experiment.metrics import (
    compute_per_condition_accuracy,
    contamination_influence_score,
    per_justice_accuracy_by_condition,
    spoiler_retrieval_analysis,
)

log = logging.getLogger(__name__)

# Style
COLORS = {
    "A": "#3b82f6",  # Blue — baseline
    "B": "#ef4444",  # Red — spoiler
    "C": "#22c55e",  # Green — anti-contamination
    "D": "#f59e0b",  # Amber — anti-contam + spoiler
}
CONDITION_LABELS = {
    "A": "Baseline",
    "B": "Spoiler",
    "C": "Anti-Contam.",
    "D": "Anti-Contam.\n+ Spoiler",
}


def _setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def figure_condition_comparison(
    results: list[dict], out_path: Path
) -> None:
    """Figure 1: Grouped bar chart — conditions × accuracy metrics."""
    _setup_style()
    acc = compute_per_condition_accuracy(results)
    conditions = sorted(acc.keys())

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(conditions))
    width = 0.35

    case_accs = [acc[c]["case_accuracy"] * 100 for c in conditions]
    justice_accs = [acc[c]["justice_accuracy"] * 100 for c in conditions]

    bars1 = ax.bar(x - width / 2, case_accs, width,
                   label="Case Outcome", color=[COLORS.get(c, "#888") for c in conditions],
                   alpha=0.9, edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, justice_accs, width,
                   label="Justice Votes", color=[COLORS.get(c, "#888") for c in conditions],
                   alpha=0.5, edgecolor="white", linewidth=0.5,
                   hatch="//")

    # Value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Experimental Condition")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Prediction Accuracy by Experimental Condition")
    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS.get(c, c) for c in conditions])
    ax.legend(loc="upper right")
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    log.info("Saved: %s", out_path)


def figure_confidence_violins(
    results: list[dict], out_path: Path
) -> None:
    """Figure 2: Confidence distribution violin plots per condition."""
    _setup_style()

    by_cond = defaultdict(list)
    for r in results:
        by_cond[r["condition"]].append(r["average_confidence"])

    conditions = sorted(by_cond.keys())
    data = [by_cond[c] for c in conditions]

    fig, ax = plt.subplots(figsize=(7, 5))

    parts = ax.violinplot(data, positions=range(len(conditions)),
                          showmeans=True, showmedians=True)

    for i, pc in enumerate(parts["bodies"]):
        c = conditions[i]
        pc.set_facecolor(COLORS.get(c, "#888"))
        pc.set_alpha(0.6)

    # Overlay individual points with jitter
    for i, c in enumerate(conditions):
        jitter = np.random.default_rng(42).normal(0, 0.04, len(by_cond[c]))
        ax.scatter(i + jitter, by_cond[c], alpha=0.3, s=10,
                   color=COLORS.get(c, "#888"), zorder=3)

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels([CONDITION_LABELS.get(c, c) for c in conditions])
    ax.set_ylabel("Average Confidence")
    ax.set_title("Confidence Distribution by Condition")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    log.info("Saved: %s", out_path)


def figure_justice_heatmap(
    results: list[dict], out_path: Path
) -> None:
    """Figure 3: Per-justice accuracy heatmap (justices × conditions)."""
    _setup_style()

    pj = per_justice_accuracy_by_condition(results)
    justices = sorted(pj.keys())
    conditions = sorted(set(c for j in pj.values() for c in j.keys()))

    matrix = np.zeros((len(justices), len(conditions)))
    for i, j in enumerate(justices):
        for k, c in enumerate(conditions):
            matrix[i, k] = pj[j].get(c, 0)

    fig, ax = plt.subplots(figsize=(6, 7))
    im = ax.imshow(matrix * 100, cmap="RdYlGn", aspect="auto",
                   vmin=50, vmax=100)

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels([CONDITION_LABELS.get(c, c) for c in conditions])
    ax.set_yticks(range(len(justices)))
    ax.set_yticklabels([j.split()[-1] for j in justices])

    # Annotate cells
    for i in range(len(justices)):
        for k in range(len(conditions)):
            val = matrix[i, k] * 100
            color = "white" if val < 65 else "black"
            ax.text(k, i, f"{val:.0f}%", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

    ax.set_title("Justice-Level Accuracy by Condition (%)")
    fig.colorbar(im, ax=ax, label="Accuracy (%)", shrink=0.8)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    log.info("Saved: %s", out_path)


def figure_cis_by_model(
    results: list[dict], out_path: Path
) -> None:
    """Figure 4: Contamination Influence Score per LLM model."""
    _setup_style()

    by_model = defaultdict(list)
    for r in results:
        by_model[r["llm_model"]].append(r)

    models = sorted(by_model.keys())
    cis_case = []
    cis_justice = []

    for model in models:
        model_results = by_model[model]
        acc = compute_per_condition_accuracy(model_results)
        if "A" in acc and "B" in acc:
            cis_case.append(contamination_influence_score(
                acc["A"]["case_accuracy"], acc["B"]["case_accuracy"]
            ))
            cis_justice.append(contamination_influence_score(
                acc["A"]["justice_accuracy"], acc["B"]["justice_accuracy"]
            ))
        else:
            cis_case.append(0)
            cis_justice.append(0)

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(models))
    width = 0.35

    ax.bar(x - width / 2, cis_case, width, label="Case CIS", color="#ef4444", alpha=0.8)
    ax.bar(x + width / 2, cis_justice, width, label="Justice CIS", color="#f59e0b", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Contamination Influence Score")
    ax.set_title("CIS by LLM Model")
    ax.legend()
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    log.info("Saved: %s", out_path)


def figure_spoiler_funnel(
    results: list[dict], out_path: Path
) -> None:
    """Figure 5: Spoiler detection funnel."""
    _setup_style()

    spoiler_results = [r for r in results if r.get("inject_spoiler")]
    if not spoiler_results:
        log.warning("No spoiler results for funnel figure")
        return

    total = len(spoiler_results)
    cited = sum(1 for r in spoiler_results if r.get("spoiler_cited_in_reasoning"))
    prediction_changed = sum(
        1 for r in spoiler_results
        if r.get("spoiler_cited_in_reasoning") and r.get("case_correct")
        and not any(  # Would have been wrong without spoiler
            r2["docket"] == r["docket"] and r2["condition"] == "A"
            and not r2["case_correct"]
            for r2 in results
        )
    )

    stages = ["Spoiler\nInjected", "Cited in\nReasoning", "Prediction\nCorrect (B)\nvs Wrong (A)"]
    values = [total, cited, prediction_changed]

    fig, ax = plt.subplots(figsize=(8, 4))

    # Horizontal funnel bars
    max_val = max(values) if max(values) > 0 else 1
    colors = ["#ef4444", "#f59e0b", "#22c55e"]

    for i, (stage, val, color) in enumerate(zip(stages, values, colors)):
        width = val / max_val * 0.8
        bar = ax.barh(len(stages) - 1 - i, width, height=0.6,
                       color=color, alpha=0.8, left=(0.8 - width) / 2)
        ax.text(0.4, len(stages) - 1 - i,
                f"{val} ({val/total*100:.0f}%)" if total > 0 else "0",
                ha="center", va="center", fontweight="bold", fontsize=12)

    ax.set_yticks(range(len(stages)))
    ax.set_yticklabels(list(reversed(stages)))
    ax.set_xlim(0, 0.8)
    ax.set_xticks([])
    ax.set_title("Spoiler Influence Funnel")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    log.info("Saved: %s", out_path)


def generate_latex_table(results: list[dict], out_path: Path) -> None:
    """Generate LaTeX summary table."""
    acc = compute_per_condition_accuracy(results)
    conditions = sorted(acc.keys())

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Training Data Contamination Experiment Results}",
        r"\label{tab:contamination}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Condition & $N$ & Case Acc. & Justice Acc. & Avg. Conf. \\",
        r"\midrule",
    ]

    for c in conditions:
        a = acc[c]
        label = {
            "A": "Baseline",
            "B": "Spoiler",
            "C": "Anti-Contamination",
            "D": "Anti-Contam. + Spoiler",
        }.get(c, c)
        lines.append(
            f"  {label} & {a['n_cases']} & "
            f"{a['case_accuracy']:.1%} & {a['justice_accuracy']:.1%} & "
            f"{a['avg_confidence']:.3f} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    out_path.write_text("\n".join(lines))
    log.info("Saved: %s", out_path)


def generate_all_figures(
    results: list[dict],
    out_dir: Path,
) -> None:
    """Generate all paper figures from experiment results."""
    out_dir.mkdir(parents=True, exist_ok=True)

    figure_condition_comparison(results, out_dir / "fig1_condition_comparison.png")
    figure_confidence_violins(results, out_dir / "fig2_confidence_violins.png")
    figure_justice_heatmap(results, out_dir / "fig3_justice_heatmap.png")
    figure_cis_by_model(results, out_dir / "fig4_cis_by_model.png")
    figure_spoiler_funnel(results, out_dir / "fig5_spoiler_funnel.png")
    generate_latex_table(results, out_dir / "table1_summary.tex")

    log.info("All figures generated in %s", out_dir)
