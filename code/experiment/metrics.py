"""Statistical analysis for contamination experiments.

Computes:
- Per-condition accuracy (case outcome + justice votes)
- Contamination Influence Score (CIS)
- Anti-Contamination Effectiveness (ACE)
- McNemar's test (paired binary outcomes)
- Paired t-test and Wilcoxon (justice accuracy)
- KS test (confidence distributions)
- Spoiler retrieval rate analysis
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# Accuracy computation
# ══════════════════════════════════════════════════════════════════════════

def compute_per_condition_accuracy(results: list[dict]) -> dict[str, dict]:
    """Group results by condition and compute accuracy metrics.

    Returns:
        {
            "A": {
                "case_accuracy": 0.66, "justice_accuracy": 0.68,
                "n_cases": 64, "avg_confidence": 0.72,
                "case_correct": 42, "case_total": 64,
                "justice_correct": 392, "justice_total": 576,
            },
            ...
        }
    """
    by_condition: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_condition[r["condition"]].append(r)

    metrics = {}
    for cond, cond_results in sorted(by_condition.items()):
        case_correct = sum(1 for r in cond_results if r["case_correct"])
        case_total = len(cond_results)
        justice_correct = sum(r.get("justice_correct", 0) for r in cond_results)
        justice_total = sum(r.get("justice_total", 0) for r in cond_results)
        confidences = [r["average_confidence"] for r in cond_results]

        metrics[cond] = {
            "case_accuracy": case_correct / case_total if case_total else 0,
            "justice_accuracy": justice_correct / justice_total if justice_total else 0,
            "n_cases": case_total,
            "case_correct": case_correct,
            "case_total": case_total,
            "justice_correct": justice_correct,
            "justice_total": justice_total,
            "avg_confidence": float(np.mean(confidences)) if confidences else 0,
            "std_confidence": float(np.std(confidences)) if confidences else 0,
        }

    return metrics


# ══════════════════════════════════════════════════════════════════════════
# Contamination scores
# ══════════════════════════════════════════════════════════════════════════

def contamination_influence_score(acc_baseline: float, acc_spoiler: float) -> float:
    """CIS = (Acc_B - Acc_A) / (1 - Acc_A)

    Measures what fraction of the remaining improvable accuracy is captured
    by the spoiler. CIS=0 means no contamination effect, CIS=1 means the
    spoiler accounts for all remaining improvement.
    """
    if acc_baseline >= 1.0:
        return 0.0
    return (acc_spoiler - acc_baseline) / (1.0 - acc_baseline)


def anti_contamination_effectiveness(
    acc_baseline: float, acc_spoiler: float, acc_spoiler_anti: float
) -> float:
    """ACE = 1 - (Acc_D - Acc_A) / (Acc_B - Acc_A)

    Measures how much of the spoiler effect is neutralized by anti-contamination
    prompting. ACE=1 means perfect neutralization, ACE=0 means no effect.
    """
    spoiler_effect = acc_spoiler - acc_baseline
    if abs(spoiler_effect) < 1e-10:
        return 1.0  # No spoiler effect to neutralize
    residual = acc_spoiler_anti - acc_baseline
    return 1.0 - residual / spoiler_effect


# ══════════════════════════════════════════════════════════════════════════
# Statistical tests
# ══════════════════════════════════════════════════════════════════════════

def _align_paired_results(
    results: list[dict], cond_a: str, cond_b: str
) -> tuple[list[dict], list[dict]]:
    """Align results from two conditions by (docket, llm_model, replicate)."""
    key_fn = lambda r: (r["docket"], r["llm_model"], r["replicate"])

    by_key_a = {key_fn(r): r for r in results if r["condition"] == cond_a}
    by_key_b = {key_fn(r): r for r in results if r["condition"] == cond_b}

    common_keys = sorted(set(by_key_a) & set(by_key_b))
    paired_a = [by_key_a[k] for k in common_keys]
    paired_b = [by_key_b[k] for k in common_keys]
    return paired_a, paired_b


def mcnemar_test(results: list[dict], cond_a: str, cond_b: str) -> dict:
    """McNemar's test for paired binary case-outcome correctness.

    Tests whether the marginal proportions of correct predictions differ
    between conditions. This is the right test for matched-pairs binary data.
    """
    paired_a, paired_b = _align_paired_results(results, cond_a, cond_b)
    if not paired_a:
        return {"error": f"No paired results for {cond_a} vs {cond_b}"}

    # Contingency: (a_correct & b_wrong), (a_wrong & b_correct)
    b_count = 0  # A correct, B wrong
    c_count = 0  # A wrong, B correct

    for a, b in zip(paired_a, paired_b):
        a_ok = a["case_correct"]
        b_ok = b["case_correct"]
        if a_ok and not b_ok:
            b_count += 1
        elif not a_ok and b_ok:
            c_count += 1

    n_discordant = b_count + c_count
    if n_discordant == 0:
        return {
            "test": "McNemar",
            "conditions": f"{cond_a} vs {cond_b}",
            "n_pairs": len(paired_a),
            "b_count": b_count,
            "c_count": c_count,
            "statistic": 0.0,
            "p_value": 1.0,
            "note": "No discordant pairs",
        }

    # Use exact binomial test for small samples
    if n_discordant < 25:
        p_value = scipy_stats.binomtest(b_count, n_discordant, 0.5).pvalue
        test_name = "McNemar (exact binomial)"
    else:
        chi2 = (abs(b_count - c_count) - 1) ** 2 / (b_count + c_count)
        p_value = 1 - scipy_stats.chi2.cdf(chi2, 1)
        test_name = "McNemar (chi-squared, continuity corrected)"

    odds_ratio = b_count / c_count if c_count > 0 else float("inf")

    return {
        "test": test_name,
        "conditions": f"{cond_a} vs {cond_b}",
        "n_pairs": len(paired_a),
        "b_count": b_count,
        "c_count": c_count,
        "n_discordant": n_discordant,
        "odds_ratio": round(odds_ratio, 3),
        "p_value": round(p_value, 6),
    }


def paired_ttest_justice_accuracy(
    results: list[dict], cond_a: str, cond_b: str
) -> dict:
    """Paired t-test on per-case justice accuracy rates."""
    paired_a, paired_b = _align_paired_results(results, cond_a, cond_b)
    if not paired_a:
        return {"error": f"No paired results for {cond_a} vs {cond_b}"}

    acc_a = np.array([r["justice_accuracy"] for r in paired_a])
    acc_b = np.array([r["justice_accuracy"] for r in paired_b])
    diff = acc_b - acc_a

    t_stat, p_value = scipy_stats.ttest_rel(acc_a, acc_b)

    # Cohen's d for paired samples
    d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0

    # Wilcoxon as non-parametric alternative
    try:
        w_stat, w_pvalue = scipy_stats.wilcoxon(diff)
    except ValueError:
        w_stat, w_pvalue = 0.0, 1.0

    return {
        "test": "Paired t-test + Wilcoxon",
        "conditions": f"{cond_a} vs {cond_b}",
        "n_pairs": len(paired_a),
        "mean_diff": round(float(np.mean(diff)), 4),
        "std_diff": round(float(np.std(diff, ddof=1)), 4),
        "t_statistic": round(float(t_stat), 4),
        "t_p_value": round(float(p_value), 6),
        "cohens_d": round(float(d), 4),
        "wilcoxon_statistic": round(float(w_stat), 4),
        "wilcoxon_p_value": round(float(w_pvalue), 6),
    }


def confidence_ks_test(results: list[dict], cond_a: str, cond_b: str) -> dict:
    """KS test comparing confidence distributions between conditions."""
    conf_a = [r["average_confidence"] for r in results if r["condition"] == cond_a]
    conf_b = [r["average_confidence"] for r in results if r["condition"] == cond_b]

    if not conf_a or not conf_b:
        return {"error": f"No data for {cond_a} or {cond_b}"}

    ks_stat, p_value = scipy_stats.ks_2samp(conf_a, conf_b)

    return {
        "test": "KS two-sample",
        "conditions": f"{cond_a} vs {cond_b}",
        "n_a": len(conf_a),
        "n_b": len(conf_b),
        "mean_a": round(float(np.mean(conf_a)), 4),
        "mean_b": round(float(np.mean(conf_b)), 4),
        "ks_statistic": round(float(ks_stat), 4),
        "p_value": round(float(p_value), 6),
    }


# ══════════════════════════════════════════════════════════════════════════
# Spoiler analysis
# ══════════════════════════════════════════════════════════════════════════

def spoiler_retrieval_analysis(results: list[dict]) -> dict:
    """Analyze spoiler retrieval rates for conditions B and D."""
    spoiler_results = [r for r in results if r.get("inject_spoiler")]
    if not spoiler_results:
        return {"error": "No spoiler condition results"}

    total = len(spoiler_results)
    cited = sum(1 for r in spoiler_results if r.get("spoiler_cited_in_reasoning"))
    cited_justices_all = []
    for r in spoiler_results:
        cited_justices_all.extend(r.get("spoiler_cited_justices", []))

    # Per-condition breakdown
    by_cond = defaultdict(list)
    for r in spoiler_results:
        by_cond[r["condition"]].append(r)

    cond_stats = {}
    for cond, cond_results in by_cond.items():
        n = len(cond_results)
        n_cited = sum(1 for r in cond_results if r.get("spoiler_cited_in_reasoning"))
        cond_stats[cond] = {
            "total": n,
            "cited_in_reasoning": n_cited,
            "citation_rate": round(n_cited / n, 3) if n > 0 else 0,
        }

    return {
        "total_spoiler_runs": total,
        "spoiler_cited_total": cited,
        "overall_citation_rate": round(cited / total, 3) if total > 0 else 0,
        "most_susceptible_justices": _count_susceptible(cited_justices_all),
        "per_condition": cond_stats,
    }


def _count_susceptible(justice_names: list[str]) -> list[tuple[str, int]]:
    """Count which justices most frequently cite the spoiler."""
    counts: dict[str, int] = defaultdict(int)
    for name in justice_names:
        counts[name] += 1
    return sorted(counts.items(), key=lambda x: x[1], reverse=True)


# ══════════════════════════════════════════════════════════════════════════
# Per-justice breakdown
# ══════════════════════════════════════════════════════════════════════════

def per_justice_accuracy_by_condition(results: list[dict]) -> dict:
    """Compute per-justice accuracy for each condition.

    Returns:
        {
            "John Roberts": {"A": 0.75, "B": 0.82, "C": 0.73, "D": 0.78},
            ...
        }
    """
    from scotus_v2.evaluate import JUSTICE_NAME_MAP

    justice_stats: dict[str, dict[str, dict]] = defaultdict(
        lambda: defaultdict(lambda: {"correct": 0, "total": 0})
    )

    for r in results:
        cond = r["condition"]
        justice_votes = r.get("justice_votes", {})
        actual_winner = r["actual_winner"]

        # We need per-justice ground truth. Use the ground truth lookup.
        gt = load_ground_truth_cached()
        gt_case = gt.get(r["docket"], {})
        gt_votes = gt_case.get("justice_votes", {})

        for full_name, abbrev in JUSTICE_NAME_MAP.items():
            if full_name in justice_votes and abbrev in gt_votes:
                justice_stats[full_name][cond]["total"] += 1
                if justice_votes[full_name]["vote"] == gt_votes[abbrev]:
                    justice_stats[full_name][cond]["correct"] += 1

    # Convert to accuracy rates
    result = {}
    for name, cond_stats in justice_stats.items():
        result[name] = {}
        for cond, s in cond_stats.items():
            result[name][cond] = round(s["correct"] / s["total"], 3) if s["total"] > 0 else 0
    return result


_gt_cache = None

def load_ground_truth_cached():
    global _gt_cache
    if _gt_cache is None:
        from scotus_v2.evaluate import load_ground_truth
        _gt_cache = load_ground_truth()
    return _gt_cache


# ══════════════════════════════════════════════════════════════════════════
# Full analysis report
# ══════════════════════════════════════════════════════════════════════════

def full_analysis(results: list[dict]) -> dict:
    """Run all analyses and return a comprehensive report."""
    report = {
        "per_condition": compute_per_condition_accuracy(results),
        "n_total_results": len(results),
    }

    # Compute contamination scores
    acc = report["per_condition"]
    if "A" in acc and "B" in acc:
        report["contamination_influence_score"] = {
            "case": round(contamination_influence_score(
                acc["A"]["case_accuracy"], acc["B"]["case_accuracy"]
            ), 4),
            "justice": round(contamination_influence_score(
                acc["A"]["justice_accuracy"], acc["B"]["justice_accuracy"]
            ), 4),
        }

    if "A" in acc and "B" in acc and "D" in acc:
        report["anti_contamination_effectiveness"] = {
            "case": round(anti_contamination_effectiveness(
                acc["A"]["case_accuracy"], acc["B"]["case_accuracy"],
                acc["D"]["case_accuracy"]
            ), 4),
            "justice": round(anti_contamination_effectiveness(
                acc["A"]["justice_accuracy"], acc["B"]["justice_accuracy"],
                acc["D"]["justice_accuracy"]
            ), 4),
        }

    # Statistical tests
    tests = {}
    condition_labels = sorted(set(r["condition"] for r in results))
    if "A" in condition_labels and "B" in condition_labels:
        tests["mcnemar_A_vs_B"] = mcnemar_test(results, "A", "B")
        tests["ttest_A_vs_B"] = paired_ttest_justice_accuracy(results, "A", "B")
        tests["ks_confidence_A_vs_B"] = confidence_ks_test(results, "A", "B")
    if "A" in condition_labels and "C" in condition_labels:
        tests["mcnemar_A_vs_C"] = mcnemar_test(results, "A", "C")
        tests["ttest_A_vs_C"] = paired_ttest_justice_accuracy(results, "A", "C")
    if "B" in condition_labels and "D" in condition_labels:
        tests["mcnemar_B_vs_D"] = mcnemar_test(results, "B", "D")
        tests["ttest_B_vs_D"] = paired_ttest_justice_accuracy(results, "B", "D")
    if "A" in condition_labels and "D" in condition_labels:
        tests["mcnemar_A_vs_D"] = mcnemar_test(results, "A", "D")
    report["statistical_tests"] = tests

    # Spoiler analysis
    report["spoiler_analysis"] = spoiler_retrieval_analysis(results)

    # Per-justice breakdown
    report["per_justice"] = per_justice_accuracy_by_condition(results)

    return report


def print_report(report: dict) -> None:
    """Print a human-readable analysis report."""
    print("\n" + "=" * 80)
    print("CONTAMINATION EXPERIMENT — ANALYSIS REPORT")
    print("=" * 80)

    # Per-condition accuracy
    print("\n── Per-Condition Accuracy ──")
    acc = report["per_condition"]
    print(f"{'Cond':<6} {'Cases':>6} {'Case Acc':>10} {'Justice Acc':>12} {'Avg Conf':>10}")
    print("-" * 50)
    for cond in sorted(acc.keys()):
        a = acc[cond]
        print(f"  {cond:<4} {a['n_cases']:>6} {a['case_accuracy']:>9.1%} {a['justice_accuracy']:>11.1%} {a['avg_confidence']:>9.3f}")

    # Contamination scores
    if "contamination_influence_score" in report:
        cis = report["contamination_influence_score"]
        print(f"\n── Contamination Influence Score (CIS) ──")
        print(f"  Case-level:    {cis['case']:.4f}")
        print(f"  Justice-level: {cis['justice']:.4f}")

    if "anti_contamination_effectiveness" in report:
        ace = report["anti_contamination_effectiveness"]
        print(f"\n── Anti-Contamination Effectiveness (ACE) ──")
        print(f"  Case-level:    {ace['case']:.4f}")
        print(f"  Justice-level: {ace['justice']:.4f}")

    # Statistical tests
    if "statistical_tests" in report:
        print(f"\n── Statistical Tests ──")
        for name, test in report["statistical_tests"].items():
            if "error" in test:
                print(f"  {name}: {test['error']}")
                continue
            p = test.get("p_value") or test.get("t_p_value", "N/A")
            print(f"  {name}: p={p}, n={test.get('n_pairs', 'N/A')}")

    # Spoiler analysis
    if "spoiler_analysis" in report and "error" not in report["spoiler_analysis"]:
        sa = report["spoiler_analysis"]
        print(f"\n── Spoiler Analysis ──")
        print(f"  Total spoiler runs: {sa['total_spoiler_runs']}")
        print(f"  Cited in reasoning: {sa['spoiler_cited_total']} ({sa['overall_citation_rate']:.1%})")
        if sa["most_susceptible_justices"]:
            print(f"  Most susceptible: {sa['most_susceptible_justices'][:3]}")

    # Per-justice heatmap (text version)
    if "per_justice" in report:
        print(f"\n── Per-Justice Accuracy by Condition ──")
        pj = report["per_justice"]
        conds = sorted(set(c for j in pj.values() for c in j.keys()))
        header = f"{'Justice':<25}" + "".join(f"{c:>8}" for c in conds)
        print(header)
        print("-" * (25 + 8 * len(conds)))
        for name in sorted(pj.keys()):
            row = f"{name:<25}"
            for c in conds:
                val = pj[name].get(c, 0)
                row += f"{val:>7.1%} "
            print(row)

    print("\n" + "=" * 80)
