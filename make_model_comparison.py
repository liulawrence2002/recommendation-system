#!/usr/bin/env python3
"""
Model-comparison figure for the deck.

The `erika model/` notebooks (f1_at_k / precision_at_k / recall_at_k) print their
results as tables but never render a chart. This script reproduces the SAME warm-
library theme used by bookrec/reports/run_eda.py and turns the Step 5 held-out
evaluation (the only non-leaky numbers) into a slide-ready figure.

Numbers are the executed Step 5 / cell-28 outputs of the notebooks:
  90/10 hold-out, random_state=6604; k tuned by CV on the TRAIN split only.
  (per-metric tuning gives UBCF k=21-22, IBCF k=71-74 — differences are negligible;
   the f1_at_k run is used as canonical, and recall_at_k matches it exactly.)

NOT used: the Step 3 / cells 15-18-30-31 numbers (UBCF P@10 0.7479 etc.). Those
fit on the FULL trainset (test rows included) and then score on the test set, so
they are optimistic by leakage and must not be reported as held-out performance.

Run:  C:/Python314/python.exe "make_model_comparison.py"
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# --- palette (identical to bookrec/reports/run_eda.py) --------------------- #
PAPER, PANEL, INK, MUTED = "#FBF8F2", "#FFFFFF", "#2B2620", "#8C8275"
TEAL, TERRA, GOLD, GRID = "#1F5C6B", "#C2683C", "#D9A441", "#E7E0D5"

_installed = {f.name for f in font_manager.fontManager.ttflist}
for _f in ("Segoe UI", "Calibri", "Arial", "DejaVu Sans"):
    if _f in _installed:
        _BODY = _f
        break
else:
    _BODY = "DejaVu Sans"

plt.rcParams.update({
    "figure.facecolor": PAPER, "savefig.facecolor": PAPER, "axes.facecolor": PANEL,
    "font.family": _BODY, "font.size": 13, "text.color": INK,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.9, "axes.spines.top": False,
    "axes.spines.right": False, "figure.dpi": 110,
})

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deliverable images")
os.makedirs(OUT_DIR, exist_ok=True)
SOURCE = ("Source: erika model notebooks, Step 5 held-out evaluation "
          "(90/10 split, seed 6604; k tuned by CV on train only)")

# --- held-out results (Step 5 / cell 28) ----------------------------------- #
MODELS = ["Baseline", "UBCF\n(pearson)", "IBCF\n(cosine)"]
COLORS = [TEAL, TERRA, GOLD]
RMSE = [0.8423, 1.0287, 0.8565]
P10  = [0.6568, 0.6586, 0.6414]
R10  = [0.7912, 0.7930, 0.7734]
F1   = [0.7178, 0.7196, 0.7012]


def header(fig, kicker, title, subtitle):
    fig.text(0.045, 0.945, kicker.upper(), color=TERRA, fontsize=12.5,
             fontweight="bold", ha="left", va="top")
    fig.text(0.045, 0.905, title, color=INK, fontsize=22, fontweight="bold",
             ha="left", va="top")
    fig.text(0.045, 0.845, subtitle, color=MUTED, fontsize=13.5, ha="left", va="top")


def footer(fig, note=None):
    fig.text(0.045, 0.025, SOURCE, color=MUTED, fontsize=9, ha="left", va="bottom")
    if note:
        fig.text(0.955, 0.025, note, color=MUTED, fontsize=9, ha="right", va="bottom")


fig = plt.figure(figsize=(12.8, 7.2))
fig.patch.set_facecolor(PAPER)
header(fig, "Model bake-off · held-out test set",
       "Baseline wins on RMSE; UBCF edges the ranking metrics",
       "Rating accuracy (RMSE, lower = better) and Top-10 ranking quality often "
       "disagree — so we report both.")

# -- left panel: RMSE -- #
ax1 = fig.add_axes([0.07, 0.14, 0.37, 0.57])
ax1.set_axisbelow(True); ax1.grid(axis="x", visible=False); ax1.tick_params(length=0)
bars = ax1.bar(MODELS, RMSE, color=COLORS, edgecolor="white", width=0.62)
for b, v in zip(bars, RMSE):
    ax1.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom",
             fontsize=12.5, fontweight="bold", color=INK)
ax1.axhline(RMSE[0], color=INK, ls="--", lw=1.3)
ax1.text(2.5, RMSE[0], "  baseline = bar to beat", color=INK, fontsize=10.5,
         va="center", ha="right")
ax1.set_ylabel("RMSE  (lower is better)")
ax1.set_ylim(0, max(RMSE) * 1.18)
ax1.set_title("Rating accuracy", fontsize=14, fontweight="bold", loc="left", pad=8)

# -- right panel: ranking metrics (grouped by metric, colored by model) -- #
ax2 = fig.add_axes([0.55, 0.14, 0.40, 0.57])
ax2.set_axisbelow(True); ax2.grid(axis="x", visible=False); ax2.tick_params(length=0)
metrics = ["Precision@10", "Recall@10", "F1@10"]
data = np.array([P10, R10, F1])          # rows = metric, cols = model
x = np.arange(len(metrics)); w = 0.26
for j, (name, c) in enumerate(zip(["Baseline", "UBCF", "IBCF"], COLORS)):
    offs = (j - 1) * w
    b = ax2.bar(x + offs, data[:, j], width=w, color=c, edgecolor="white", label=name)
    for rect, v in zip(b, data[:, j]):
        ax2.text(rect.get_x() + rect.get_width() / 2, v, f"{v:.2f}", ha="center",
                 va="bottom", fontsize=9.5, color=INK)
ax2.set_xticks(x); ax2.set_xticklabels(metrics)
ax2.set_ylabel("Score  (higher is better)")
ax2.set_ylim(0, 1.0)
ax2.set_title("Top-10 ranking quality", fontsize=14, fontweight="bold", loc="left", pad=8)
ax2.legend(frameon=False, fontsize=11, ncol=3, loc="upper center",
           bbox_to_anchor=(0.5, 1.16))

footer(fig, "relevant = held-out rating ≥ 4")
path = os.path.join(OUT_DIR, "model_comparison.png")
fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.35, facecolor=PAPER)
plt.close(fig)
print("saved", path)
