"""Where the analysis scripts find their inputs, in either checkout layout.

  GitHub repo (JohKrus/Scotus_Prediction):
      <repo>/analysis/*.py             these scripts
      <repo>/predictions/              pipeline outputs (term_*, ot2025_reruns, ...)
      <repo>/analysis/data/            SCDB CSVs (downloaded separately) and OT2025 results
  Working-paper folder:
      <root>/v2/code/*.py              these scripts
      <root>/v2/predictions/           pipeline outputs
      <root>/                          SCDB CSVs and OT2025 results sheet
"""
from __future__ import annotations

import os

HERE = os.path.dirname(os.path.abspath(__file__))

if os.path.basename(HERE) == "analysis":
    ROOT = os.path.dirname(HERE)
    PRED_DIR = os.path.join(ROOT, "predictions")
    DATA_DIR = os.path.join(HERE, "data")
else:
    ROOT = os.path.dirname(os.path.dirname(HERE))
    PRED_DIR = os.path.join(ROOT, "v2", "predictions")
    DATA_DIR = ROOT

PAPER_DIR = os.path.join(ROOT, "paper")
