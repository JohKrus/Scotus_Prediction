"""Regenerate paper figures for the deliberation pipeline (v2), scored vs SCDB.

Produces, in paper/images/:
  v2_accuracy_by_term.png   grouped bar chart, case accuracy by Term x model + baseline
  v2_cm_case_winner.png     case-outcome confusion matrices (per model, pooled scoreable Terms)
  v2_heatmap_justice.png     justice x model justice-level accuracy heatmap
  v2_calibration.png         reliability curve for justice-vote confidence

Run from repo root:  python v2/code/make_figures.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import PAPER_DIR  # noqa: E402
from score_predictions import load_scdb, JUSTICE_MAP, wilson, PRED_GLOB  # noqa: E402

IMG = os.path.join(PAPER_DIR, "images")
MODELS = ["GPT-5.2", "Claude-4.6", "Gemini-2.5"]
SCOREABLE = (2022, 2023, 2024)
JUSTICES = list(JUSTICE_MAP.keys())


def collect():
    case_winner, justice_vote, base = load_scdb()
    case_term = load_scdb.case_term
    # case-level: (term, model) -> list[(pred, truth)]
    case = defaultdict(list)
    # justice-level: (model, justice) -> [correct, n]  pooled over scoreable terms
    jacc = defaultdict(lambda: [0, 0])
    calib = defaultdict(lambda: [0, 0])
    for f in glob.glob(PRED_GLOB):
        try:
            arr = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        for d in arr:
            docket = d.get("docket"); model = d.get("llm_model")
            truth = case_winner.get(docket); term = case_term.get(docket)
            if truth is None or term is None:
                continue
            case[(term, model)].append((d.get("predicted_winner"), truth))
            for jname, jv in d.get("justice_votes", {}).items():
                tv = justice_vote.get((docket, JUSTICE_MAP.get(jname)))
                if tv is None:
                    continue
                c = int(jv.get("vote") == tv)
                jacc[(model, jname)][0] += c; jacc[(model, jname)][1] += 1
                conf = jv.get("confidence")
                if isinstance(conf, (int, float)):
                    calib[min(int(conf * 10), 9)][0] += c
                    calib[min(int(conf * 10), 9)][1] += 1
    return case, jacc, calib, base


def fig_accuracy_by_term(case, base):
    terms = SCOREABLE
    x = np.arange(len(terms)); w = 0.26
    colors = {"GPT-5.2": "#4C72B0", "Claude-4.6": "#DD8452", "Gemini-2.5": "#55A868"}
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for i, m in enumerate(MODELS):
        pts, los, his = [], [], []
        for t in terms:
            pairs = case.get((t, m), [])
            k = sum(p == tr for p, tr in pairs); n = len(pairs)
            if n:
                p, lo, hi = wilson(k, n)
            else:
                p = lo = hi = np.nan
            pts.append(p); los.append(p - lo if n else 0); his.append(hi - p if n else 0)
        ax.bar(x + (i - 1) * w, pts, w, label=m, color=colors[m],
               yerr=[los, his], capsize=3, error_kw={"elinewidth": 1})
    # baseline markers per term
    for j, t in enumerate(terms):
        b = 100 * base[t]
        ax.hlines(b, x[j] - 1.5 * w, x[j] + 1.5 * w, colors="black",
                  linestyles="--", linewidth=1.3,
                  label="Always-petitioner baseline" if j == 0 else None)
    ax.set_xticks(x); ax.set_xticklabels([f"OT{t}" for t in terms])
    ax.set_ylabel("Case-outcome accuracy (%)"); ax.set_ylim(50, 100)
    ax.set_title("Case-outcome accuracy by Term (deliberation pipeline)")
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(IMG, "v2_accuracy_by_term.png"), dpi=200)
    plt.close(fig)


def fig_confusion(case):
    # per model, pooled over scoreable terms
    labels = ["Petitioner", "Respondent"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    for ax, m in zip(axes, MODELS):
        cm = np.zeros((2, 2), dtype=int)  # rows actual, cols predicted
        for t in SCOREABLE:
            for pred, truth in case.get((t, m), []):
                if pred in labels and truth in labels:
                    cm[labels.index(truth), labels.index(pred)] += 1
        n = cm.sum(); acc = 100 * np.trace(cm) / n if n else 0
        im = ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for k in range(2):
                ax.text(k, i, str(cm[i, k]), ha="center", va="center",
                        color="white" if cm[i, k] > cm.max() / 2 else "black", fontsize=12)
        ax.set_xticks([0, 1]); ax.set_xticklabels(labels, fontsize=8)
        ax.set_yticks([0, 1]); ax.set_yticklabels(labels, fontsize=8, rotation=90, va="center")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(f"{m} ({acc:.0f}%)")
    fig.suptitle("Case-outcome confusion matrices, pooled over OT2022–OT2024", y=1.02)
    fig.tight_layout(); fig.savefig(os.path.join(IMG, "v2_cm_case_winner.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_heatmap(jacc):
    mat = np.full((len(JUSTICES), len(MODELS)), np.nan)
    for r, j in enumerate(JUSTICES):
        for c, m in enumerate(MODELS):
            k, n = jacc.get((m, j), [0, 0])
            if n:
                mat[r, c] = 100 * k / n
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    im = ax.imshow(mat, cmap="viridis", vmin=60, vmax=85, aspect="auto")
    ax.set_xticks(range(len(MODELS))); ax.set_xticklabels(MODELS, fontsize=9)
    short = [j.split()[-1] if j != "Ketanji Brown Jackson" else "Jackson" for j in JUSTICES]
    ax.set_yticks(range(len(JUSTICES))); ax.set_yticklabels(short, fontsize=9)
    for r in range(len(JUSTICES)):
        for c in range(len(MODELS)):
            if not np.isnan(mat[r, c]):
                ax.text(c, r, f"{mat[r, c]:.0f}", ha="center", va="center",
                        color="white" if mat[r, c] < 74 else "black", fontsize=9)
    ax.set_title("Justice-level accuracy (%), pooled OT2022–OT2024")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Accuracy (%)")
    fig.tight_layout(); fig.savefig(os.path.join(IMG, "v2_heatmap_justice.png"), dpi=200)
    plt.close(fig)


def fig_calibration(calib):
    xs, ys, sizes = [], [], []
    for b in range(10):
        k, n = calib.get(b, [0, 0])
        if n >= 10:  # drop noise bins (matches the paper table, which starts at [0.5,0.6))
            xs.append((b + 0.5) / 10); ys.append(k / n); sizes.append(n)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1.2, label="perfect calibration")
    s = np.array(sizes); s = 40 + 360 * (s / s.max())
    ax.scatter(xs, ys, s=s, color="#4C72B0", alpha=0.8, edgecolor="black", zorder=3)
    ax.set_xlabel("Stated confidence (bin midpoint)")
    ax.set_ylabel("Empirical accuracy")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Calibration of justice-vote confidence")
    ax.legend(fontsize=9, loc="upper left"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(IMG, "v2_calibration.png"), dpi=200)
    plt.close(fig)


def main():
    case, jacc, calib, base = collect()
    fig_accuracy_by_term(case, base)
    fig_confusion(case)
    fig_heatmap(jacc)
    fig_calibration(calib)
    print("Wrote: v2_accuracy_by_term.png, v2_cm_case_winner.png, "
          "v2_heatmap_justice.png, v2_calibration.png to", IMG)


if __name__ == "__main__":
    main()
