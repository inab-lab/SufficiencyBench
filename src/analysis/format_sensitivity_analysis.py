"""
format_sensitivity_analysis.py — Cross-format probe transfer.

Trains CSP probes on SufficiencyBench (standard short-passage format, Paper 1)
and evaluates on DiverseBench test examples grouped by context-presentation
format (dpr_chunk, article_intro, long_passage, wrong_article, short_passage).

This tests whether the sufficiency direction in the residual stream is
invariant to how the retrieved context is structured.

Saves:
  results/format_sensitivity.json
  results/figures/format_sensitivity.pdf

Usage:
  python src/analysis/format_sensitivity_analysis.py
"""

import gc
import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.paths import HIDDEN_STATES_DIR, RESULTS_DIR, FIGURES_DIR

plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "font.size":       11,
    "axes.titlesize":  12,
    "axes.labelsize":  12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})

# Diverse bench paths (Paper 2)
PAPER2_ROOT = Path(__file__).resolve().parents[2].parent / "knowledge_state_geometry"
DIVERSE_HS   = PAPER2_ROOT / "data" / "diverse_bench" / "hidden_states"

# Best layers (validation-selected in Paper 1)
BEST_LAYERS = {"llama": 12, "mistral": 12, "qwen": 16}
