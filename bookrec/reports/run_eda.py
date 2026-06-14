#!/usr/bin/env python3
"""
Project 2 — Executive + Technical EDA generator.

Loads the catalog through the package `data_loader` (same [user_id, book_id, rating]
contract used everywhere else), reproduces the deep-dive EDA from
`notebooks/02_eda_deep_dive.ipynb`, and layers on the analyses a senior
recommender-systems data scientist would expect before defending a modeling
choice:

  * user-item matrix sparsity / density
  * popularity concentration as a Lorenz curve + Gini coefficient
  * Zipf / power-law fit of the long tail (log-log slope)
  * additive rating-bias decomposition (global + user + item), with in-sample R^2
  * empirical-Bayes (Bayesian shrinkage) of book averages vs raw averages
  * reader engagement segmentation (whales vs. minnows)
  * calibration of the sample's observed averages against the global Goodreads avg
  * catalog vintage + metadata completeness

Outputs (all under bookrec/reports/):
  figures/exec_*.png        slide-ready executive charts (16:9, 200 dpi)
  figures/appendix_*.png    slide-ready technical-appendix charts
  eda_stats.json            every headline number, machine-readable
  eda_tables/*.csv          the underlying tables
  eda_summary.md            narrative report (exec summary + technical appendix)

Run from anywhere:
    C:/Python314/python.exe bookrec/reports/run_eda.py
"""
from __future__ import annotations

import json
import os
import sys
import textwrap

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

# --------------------------------------------------------------------------- #
# Paths — resolve relative to this file so the script runs from any CWD.
# --------------------------------------------------------------------------- #
REPORTS_DIR = os.path.dirname(os.path.abspath(__file__))
BOOKREC_DIR = os.path.dirname(REPORTS_DIR)
DATA_DIR = os.path.join(BOOKREC_DIR, "data")
FIG_DIR = os.path.join(REPORTS_DIR, "figures")
TBL_DIR = os.path.join(REPORTS_DIR, "eda_tables")
sys.path.insert(0, BOOKREC_DIR)

from src import data_loader  # noqa: E402

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TBL_DIR, exist_ok=True)

# --------------------------------------------------------------------------- #
# Theme — warm-library palette (on brand with the app) tuned for projector
# legibility: high-contrast ink on warm paper, one teal workhorse + terracotta
# highlight, generous type. 16:9 canvas so charts drop straight into slides.
# --------------------------------------------------------------------------- #
PAPER   = "#FBF8F2"   # warm off-white figure background
PANEL   = "#FFFFFF"   # plotting-area background
INK     = "#2B2620"   # warm near-black for text
MUTED   = "#8C8275"   # secondary text / gridlines
TEAL    = "#1F5C6B"   # primary series
TEAL_LT = "#7FA8B2"
TERRA   = "#C2683C"   # highlight / "watch this" series
GOLD    = "#D9A441"
SAGE    = "#6E8B62"
PLUM    = "#7A5C7E"
GRID    = "#E7E0D5"
SEQ = [TEAL, TERRA, GOLD, SAGE, PLUM, TEAL_LT]

# Prefer a clean Windows UI font; fall back gracefully.
_installed = {f.name for f in font_manager.fontManager.ttflist}
for _f in ("Segoe UI", "Calibri", "Arial", "DejaVu Sans"):
    if _f in _installed:
        _BODY = _f
        break
else:
    _BODY = "DejaVu Sans"

plt.rcParams.update({
    "figure.facecolor": PAPER,
    "savefig.facecolor": PAPER,
    "axes.facecolor": PANEL,
    "font.family": _BODY,
    "font.size": 13,
    "text.color": INK,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 110,
})

FIGSIZE = (12.8, 7.2)   # 16:9
SOURCE = "Source: Project 2 ratings sample (Goodreads-derived) · loaded via bookrec.data_loader"
_saved: list[str] = []


def styled_fig():
    fig = plt.figure(figsize=FIGSIZE)
    fig.patch.set_facecolor(PAPER)
    return fig


def header(fig, kicker, title, subtitle=None):
    """Slide-style header band: small accent kicker, big title, muted dek."""
    fig.text(0.045, 0.945, kicker.upper(), color=TERRA, fontsize=12.5,
             fontweight="bold", ha="left", va="top")
    fig.text(0.045, 0.905, title, color=INK, fontsize=22, fontweight="bold",
             ha="left", va="top")
    if subtitle:
        fig.text(0.045, 0.845, subtitle, color=MUTED, fontsize=14,
                 ha="left", va="top")


def footer(fig, note=None):
    fig.text(0.045, 0.025, SOURCE, color=MUTED, fontsize=9, ha="left", va="bottom")
    if note:
        fig.text(0.955, 0.025, note, color=MUTED, fontsize=9, ha="right", va="bottom")


def save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.35,
                facecolor=PAPER)
    plt.close(fig)
    _saved.append(name)
    print(f"  saved figures/{name}")


def thousands(x, _pos=None):
    return f"{x:,.0f}"


def style_ax(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)
    ax.tick_params(length=0)
    return ax


# --------------------------------------------------------------------------- #
# Stat helpers
# --------------------------------------------------------------------------- #
def gini(values) -> float:
    """Gini coefficient of a non-negative distribution (0 = equal, 1 = max)."""
    x = np.sort(np.asarray(values, dtype=float))
    n = x.size
    if n == 0 or x.sum() == 0:
        return float("nan")
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def lorenz_points(values):
    """Return (cum_pop_fraction, cum_value_fraction) for a Lorenz curve."""
    x = np.sort(np.asarray(values, dtype=float))
    cum = np.cumsum(x) / x.sum()
    cum = np.insert(cum, 0, 0.0)
    pop = np.linspace(0.0, 1.0, cum.size)
    return pop, cum


# --------------------------------------------------------------------------- #
# Load + core frames
# --------------------------------------------------------------------------- #
print("Loading data ...")
ratings, books = data_loader.load(DATA_DIR)
print(f"  books={books.shape}  ratings={ratings.shape}")

stats: dict = {}

n_ratings = int(len(ratings))
n_users = int(ratings["user_id"].nunique())
n_books_rated = int(ratings["book_id"].nunique())
n_catalog = int(books["book_id"].nunique())
density = n_ratings / (n_users * n_books_rated)
mu = float(ratings["rating"].mean())

stats["overview"] = {
    "n_ratings": n_ratings,
    "n_users": n_users,
    "n_books_rated": n_books_rated,
    "n_catalog": n_catalog,
    "n_books_never_rated": int(n_catalog - n_books_rated),
    "matrix_cells": int(n_users * n_books_rated),
    "density_pct": round(density * 100, 4),
    "sparsity_pct": round((1 - density) * 100, 4),
    "global_mean_rating": round(mu, 4),
    "median_rating": float(ratings["rating"].median()),
    "pct_4_or_5_star": round(float((ratings["rating"] >= 4).mean()) * 100, 2),
    "avg_ratings_per_user": round(n_ratings / n_users, 2),
    "avg_ratings_per_book": round(n_ratings / n_books_rated, 2),
}

user_summary = (ratings.groupby("user_id")
                .agg(n_ratings=("rating", "count"),
                     avg_rating=("rating", "mean"),
                     rating_std=("rating", "std"),
                     unique_ratings=("rating", "nunique"))
                .reset_index())

book_summary = (ratings.groupby("book_id")
                .agg(observed_ratings=("rating", "count"),
                     observed_avg=("rating", "mean"),
                     observed_std=("rating", "std"))
                .reset_index()
                .merge(books[["book_id", "title", "authors",
                              "original_publication_year", "language_code",
                              "average_rating", "ratings_count"]],
                       on="book_id", how="left"))

book_pop = ratings["book_id"].value_counts()

# =========================================================================== #
# DATA QUALITY (reproduced from notebook 02)
# =========================================================================== #
quality = pd.DataFrame({
    "check": ["Duplicate book IDs", "Duplicate user-book ratings",
              "Ratings with no matching book", "Missing book titles",
              "Missing user IDs", "Missing book IDs in ratings",
              "Missing ratings", "Ratings outside 1-5"],
    "count": [int(books["book_id"].duplicated().sum()),
              int(ratings.duplicated(["user_id", "book_id"]).sum()),
              int((~ratings["book_id"].isin(books["book_id"])).sum()),
              int(books["title"].isna().sum()),
              int(ratings["user_id"].isna().sum()),
              int(ratings["book_id"].isna().sum()),
              int(ratings["rating"].isna().sum()),
              int((~ratings["rating"].between(1, 5)).sum())],
})
quality.to_csv(os.path.join(TBL_DIR, "data_quality.csv"), index=False)
stats["data_quality"] = dict(zip(quality["check"], quality["count"].astype(int)))

# =========================================================================== #
# COLD START
# =========================================================================== #
cold_user_thresholds = [1, 5, 10, 20]
stats["cold_start_users"] = {
    f"users_with_lt_{t}": int((user_summary["n_ratings"] < t).sum())
    for t in cold_user_thresholds
}
stats["cold_start_users"]["pct_lt_5"] = round(
    float((user_summary["n_ratings"] < 5).mean()) * 100, 2)
stats["cold_start_books"] = {
    f"books_with_lt_{t}": int((book_summary["observed_ratings"] < t).sum())
    for t in [5, 10, 20, 50]
}
stats["single_value_users"] = int((user_summary["unique_ratings"] == 1).sum())
stats["sample_design"] = {
    "min_ratings_per_user": int(user_summary["n_ratings"].min()),
    "max_ratings_per_user": int(user_summary["n_ratings"].max()),
    "median_ratings_per_user": float(user_summary["n_ratings"].median()),
    "max_ratings_per_book": int(book_summary["observed_ratings"].max()),
    "pct_books_lt_5": round(float((book_summary["observed_ratings"] < 5).mean()) * 100, 1),
    "pct_books_lt_10": round(float((book_summary["observed_ratings"] < 10).mean()) * 100, 1),
    "note": ("Every reader has 100+ ratings (min observed = "
             f"{int(user_summary['n_ratings'].min())}); this is a filtered dense-user "
             "core, so user-side cold-start is absent by construction and the cold-start "
             "risk sits on the item side."),
}

# =========================================================================== #
# CONCENTRATION: Gini + Lorenz + Zipf
# =========================================================================== #
gini_pop = gini(book_pop.values)
gini_user = gini(user_summary["n_ratings"].values)
top_share = {n: round(100 * book_pop.head(n).sum() / n_ratings, 2)
             for n in [10, 50, 100, 500, 1000]}
top_pct_share = {
    "top_1pct_books": round(100 * book_pop.head(max(1, n_books_rated // 100)).sum() / n_ratings, 2),
    "top_5pct_books": round(100 * book_pop.head(max(1, n_books_rated // 20)).sum() / n_ratings, 2),
    "top_10pct_books": round(100 * book_pop.head(max(1, n_books_rated // 10)).sum() / n_ratings, 2),
}
# Zipf / power-law slope on rank-frequency (log-log).
ranks = np.arange(1, len(book_pop) + 1)
counts_sorted = book_pop.values.astype(float)
logx, logy = np.log10(ranks), np.log10(counts_sorted)
zipf_slope, zipf_intercept = np.polyfit(logx, logy, 1)
zipf_r2 = float(1 - np.sum((logy - (zipf_slope * logx + zipf_intercept)) ** 2)
                / np.sum((logy - logy.mean()) ** 2))
stats["concentration"] = {
    "gini_book_popularity": round(gini_pop, 4),
    "gini_user_activity": round(gini_user, 4),
    "top_n_books_share_pct": top_share,
    "top_pct_books_share_pct": top_pct_share,
    "zipf_slope": round(float(zipf_slope), 3),
    "zipf_loglog_r2": round(zipf_r2, 3),
}

# =========================================================================== #
# RATING-BIAS DECOMPOSITION  (r ~= mu + b_user + b_item)
# In-sample, illustrative: shows how much rating variance a simple bias
# baseline explains -> why "predict high" is a strong RMSE competitor.
# =========================================================================== #
LAMBDA = 10.0  # regularization (prior strength) for bias estimates
gb_user = ratings.groupby("user_id")["rating"]
b_user = (gb_user.sum() - mu * gb_user.count()) / (gb_user.count() + LAMBDA)
r2 = ratings.merge(b_user.rename("b_user"), on="user_id", how="left")
# item bias computed on user-debiased residuals
r2["resid_u"] = r2["rating"] - mu - r2["b_user"]
gb_item = r2.groupby("book_id")["resid_u"]
b_item = gb_item.sum() / (gb_item.count() + LAMBDA)
r2 = r2.merge(b_item.rename("b_item"), on="book_id", how="left")

ss_tot = float(np.sum((ratings["rating"] - mu) ** 2))
pred_u = mu + r2["b_user"]
pred_ui = mu + r2["b_user"] + r2["b_item"]
r2_user = 1 - float(np.sum((r2["rating"] - pred_u) ** 2)) / ss_tot
r2_useritem = 1 - float(np.sum((r2["rating"] - pred_ui) ** 2)) / ss_tot
rmse_global = float(np.sqrt(np.mean((ratings["rating"] - mu) ** 2)))
rmse_user = float(np.sqrt(np.mean((r2["rating"] - pred_u) ** 2)))
rmse_useritem = float(np.sqrt(np.mean((r2["rating"] - pred_ui) ** 2)))
stats["bias_decomposition"] = {
    "lambda": LAMBDA,
    "global_mean": round(mu, 4),
    "r2_user_bias": round(r2_user, 4),
    "r2_user_plus_item_bias": round(r2_useritem, 4),
    "rmse_global_mean": round(rmse_global, 4),
    "rmse_user_bias": round(rmse_user, 4),
    "rmse_user_item_bias": round(rmse_useritem, 4),
    "std_user_bias": round(float(b_user.std()), 4),
    "std_item_bias": round(float(b_item.std()), 4),
    "note": "In-sample / illustrative; not a held-out evaluation.",
}

# =========================================================================== #
# EMPIRICAL-BAYES SHRINKAGE of book averages
# shrunk = (n*obs + C*mu) / (n + C),  C = prior strength (median support)
# =========================================================================== #
C = float(book_summary["observed_ratings"].median())
book_summary["shrunk_avg"] = (
    (book_summary["observed_ratings"] * book_summary["observed_avg"] + C * mu)
    / (book_summary["observed_ratings"] + C))
# How much do top-10 leaderboards differ raw vs shrunk?
top_raw = set(book_summary.sort_values("observed_avg", ascending=False).head(10)["book_id"])
top_shrunk = set(book_summary.sort_values("shrunk_avg", ascending=False).head(10)["book_id"])
stats["shrinkage"] = {
    "prior_strength_C": round(C, 2),
    "top10_overlap_raw_vs_shrunk": int(len(top_raw & top_shrunk)),
    "max_abs_shift": round(float((book_summary["observed_avg"]
                                  - book_summary["shrunk_avg"]).abs().max()), 3),
}
# Calibration vs global Goodreads average_rating (where available, >0).
cal = book_summary[(book_summary["average_rating"] > 0)
                   & book_summary["observed_avg"].notna()]
if len(cal) > 3:
    corr = float(np.corrcoef(cal["observed_avg"], cal["average_rating"])[0, 1])
else:
    corr = float("nan")
stats["calibration_observed_vs_goodreads_pearson_r"] = round(corr, 3)

# =========================================================================== #
# ENGAGEMENT SEGMENTATION (whales vs minnows)
# =========================================================================== #
us = user_summary.sort_values("n_ratings", ascending=False).reset_index(drop=True)
us["cum_share"] = us["n_ratings"].cumsum() / us["n_ratings"].sum()
seg_edges = [("Top 1%", 0.01), ("Top 5%", 0.05), ("Top 10%", 0.10),
             ("Top 25%", 0.25), ("Top 50%", 0.50)]
segmentation = {}
for label, frac in seg_edges:
    k = max(1, int(np.ceil(frac * len(us))))
    segmentation[label] = round(100 * us["n_ratings"].head(k).sum()
                                / us["n_ratings"].sum(), 2)
stats["engagement_segmentation_pct_of_ratings"] = segmentation

# =========================================================================== #
# METADATA / VINTAGE
# =========================================================================== #
meta_cols = ["language_code", "isbn", "original_publication_year", "authors"]
meta_missing = {c: int(books[c].isna().sum()) if c in books else None
                for c in meta_cols}
stats["metadata_missing_pct"] = {
    c: (round(100 * v / len(books), 2) if v is not None else None)
    for c, v in meta_missing.items()}
yr = books.loc[books["original_publication_year"] > 0, "original_publication_year"]
stats["vintage"] = {
    "median_year": float(yr.median()) if len(yr) else None,
    "pct_since_2000": round(float((yr >= 2000).mean()) * 100, 2) if len(yr) else None,
    "anomalies_year_lt0_or_gt2026": int(((books["original_publication_year"] < 0) |
                                         (books["original_publication_year"] > 2026)).sum()),
}

# Persist tables for the appendix.
user_summary.describe().to_csv(os.path.join(TBL_DIR, "user_summary_describe.csv"))
book_summary.select_dtypes("number").describe().to_csv(os.path.join(TBL_DIR, "book_summary_describe.csv"))
book_summary.sort_values("observed_ratings", ascending=False).head(20).to_csv(
    os.path.join(TBL_DIR, "top20_most_rated.csv"), index=False)

print("Computed all statistics. Rendering figures ...")

# =========================================================================== #
# ======================  EXECUTIVE FIGURES  ================================ #
# =========================================================================== #

# ---- E1. KPI scorecard --------------------------------------------------- #
def kpi_card(ax, value, label, accent):
    ax.axis("off")
    box = FancyBboxPatch((0.04, 0.08), 0.92, 0.84,
                         boxstyle="round,pad=0.02,rounding_size=0.06",
                         linewidth=0, facecolor=PANEL, mutation_aspect=0.5)
    ax.add_patch(box)
    ax.add_patch(plt.Rectangle((0.04, 0.08), 0.05, 0.84, color=accent, zorder=3))
    ax.text(0.16, 0.60, value, fontsize=27, fontweight="bold", color=INK,
            ha="left", va="center")
    ax.text(0.16, 0.26, label, fontsize=12.5, color=MUTED, ha="left", va="center")

fig = styled_fig()
header(fig, "Project 2 · Data at a glance",
       "A sparse, positively-skewed, blockbuster-driven catalog",
       "Six numbers that frame every modeling decision that follows.")
cards = [
    (f"{n_ratings:,}", "Ratings in sample", TEAL),
    (f"{n_users:,}", "Unique readers", TEAL),
    (f"{n_catalog:,}", "Books in catalog", TEAL),
    (f"{density*100:.3f}%", "User-item matrix density", TERRA),
    (f"{mu:.2f} / 5", f"Mean rating · {stats['overview']['pct_4_or_5_star']:.0f}% are 4-5 stars", TERRA),
    (f"{stats['sample_design']['pct_books_lt_10']:.0f}%", "Books with <10 ratings (item cold start)", TERRA),
]
positions = [(0.045, 0.46), (0.365, 0.46), (0.685, 0.46),
             (0.045, 0.12), (0.365, 0.12), (0.685, 0.12)]
for (val, lab, acc), (x, y) in zip(cards, positions):
    ax = fig.add_axes([x, y, 0.27, 0.30])
    kpi_card(ax, val, lab, acc)
footer(fig)
save(fig, "exec_01_kpi_scorecard.png")

# ---- E2. Long tail / Pareto ---------------------------------------------- #
fig = styled_fig()
header(fig, "The discovery problem",
       f"{top_pct_share['top_1pct_books']:.0f}% of all ratings land on the top 1% of books",
       "A naive recommender keeps surfacing the same blockbusters — why we track catalog "
       "coverage, not just accuracy.")
ax = fig.add_axes([0.075, 0.13, 0.86, 0.60])
style_ax(ax)
share = book_pop.values.cumsum() / n_ratings
xpct = np.arange(1, len(share) + 1) / len(share) * 100
ax.plot(xpct, share * 100, color=TEAL, lw=2.6)
ax.fill_between(xpct, share * 100, color=TEAL, alpha=0.08)
for marker, lbl in [(1, "1%"), (5, "5%"), (10, "10%")]:
    yval = top_pct_share[f"top_{marker}pct_books"]
    ax.scatter([marker], [yval], color=TERRA, zorder=5, s=55)
    ax.annotate(f"top {lbl} of books\n→ {yval:.0f}% of ratings",
                (marker, yval), textcoords="offset points", xytext=(14, -6),
                color=INK, fontsize=11.5, fontweight="bold", va="top")
ax.set_xlabel("Books ranked by popularity (cumulative %)")
ax.set_ylabel("Cumulative share of ratings (%)")
ax.set_xlim(0, 100)
ax.set_ylim(0, 101)
footer(fig, f"Gini = {gini_pop:.2f} (popularity inequality)")
save(fig, "exec_02_long_tail.png")

# ---- E3. Rating skew ----------------------------------------------------- #
rating_counts = ratings["rating"].value_counts().sort_index()
fig = styled_fig()
header(fig, "Ratings are not neutral feedback",
       f"{stats['overview']['pct_4_or_5_star']:.0f}% of ratings are 4 or 5 stars",
       "People rate what they already expected to like (missing-not-at-random). "
       "RMSE rewards “predict high” — so a popularity baseline is a serious benchmark.")
ax = fig.add_axes([0.075, 0.14, 0.86, 0.58])
style_ax(ax)
colors = [TERRA if r >= 4 else TEAL_LT for r in rating_counts.index]
bars = ax.bar(rating_counts.index, rating_counts.values, color=colors,
              edgecolor="white", width=0.72)
for b, v in zip(bars, rating_counts.values):
    ax.text(b.get_x() + b.get_width() / 2, v, f"{v/n_ratings*100:.0f}%",
            ha="center", va="bottom", fontsize=12, color=INK, fontweight="bold")
ax.axvline(mu, color=INK, ls="--", lw=1.4)
ax.text(mu, ax.get_ylim()[1] * 0.94, f"  mean {mu:.2f}", color=INK, fontsize=11.5)
ax.yaxis.set_major_formatter(thousands)
ax.set_xlabel("Star rating")
ax.set_ylabel("Number of ratings")
ax.set_xticks([1, 2, 3, 4, 5])
footer(fig)
save(fig, "exec_03_rating_skew.png")

# ---- E4. Cold start (item side) ------------------------------------------ #
fig = styled_fig()
header(fig, "The cold-start risk is on the books, not the readers",
       f"{stats['cold_start_books']['books_with_lt_10']:,} of {n_books_rated:,} rated books "
       f"have fewer than 10 ratings",
       "Every reader here has 100+ ratings (a filtered core), so thin signal lives on the "
       "catalog side — plus 735 books with no ratings at all.")
ax = fig.add_axes([0.075, 0.13, 0.86, 0.60])
style_ax(ax)
clip = 100
vals = book_summary["observed_ratings"].clip(upper=clip)
ax.hist(vals, bins=np.arange(1, clip + 3, 2) - 0.5, color=TEAL,
        edgecolor="white", alpha=0.92)
ax.axvspan(0.5, 9.5, color=TERRA, alpha=0.14)
ax.text(12, ax.get_ylim()[1] * 0.9,
        f"item cold-start zone\n<10 ratings: {stats['cold_start_books']['books_with_lt_10']:,} "
        f"books ({stats['sample_design']['pct_books_lt_10']:.0f}%)\n"
        f"+ {stats['overview']['n_books_never_rated']:,} books never rated",
        color=TERRA, fontsize=12.5, fontweight="bold", va="top")
ax.set_xlabel("Ratings per book")
ax.set_ylabel("Number of books")
ax.yaxis.set_major_formatter(thousands)
footer(fig, f"Median {book_summary['observed_ratings'].median():.0f} · "
            f"mean {n_ratings/n_books_rated:.0f} ratings per book")
save(fig, "exec_04_cold_start.png")

# =========================================================================== #
# ======================  APPENDIX FIGURES  ================================= #
# =========================================================================== #

# ---- A1. Matrix sparsity ------------------------------------------------- #
fig = styled_fig()
header(fig, "Appendix · A1 — Matrix sparsity",
       f"The user-item matrix is {stats['overview']['sparsity_pct']:.2f}% empty",
       "Observed cells among the 200 most-active readers × 200 most-rated books "
       "(densest corner of the matrix).")
top_u = user_summary.sort_values("n_ratings", ascending=False).head(200)["user_id"]
top_b = book_pop.head(200).index
sub = ratings[ratings["user_id"].isin(top_u) & ratings["book_id"].isin(top_b)]
uidx = {u: i for i, u in enumerate(top_u)}
bidx = {b: i for i, b in enumerate(top_b)}
M = np.zeros((len(top_u), len(top_b)))
for u, b in zip(sub["user_id"], sub["book_id"]):
    M[uidx[u], bidx[b]] = 1
ax = fig.add_axes([0.075, 0.12, 0.7, 0.62])
ax.imshow(M, aspect="auto", cmap="bone_r", interpolation="nearest")
ax.grid(False)
ax.set_xlabel("Top 200 books →")
ax.set_ylabel("Top 200 readers →")
ax.set_xticks([]); ax.set_yticks([])
dens_corner = M.mean() * 100
fig.text(0.80, 0.55,
         f"Even in this\ndensest corner,\nonly {dens_corner:.0f}% of\ncells are filled.\n\n"
         f"Full matrix:\n{n_users:,} × {n_books_rated:,}\n= {n_users*n_books_rated/1e6:.1f}M cells\n"
         f"density {density*100:.3f}%",
         fontsize=12.5, color=INK, va="center")
footer(fig)
save(fig, "appendix_01_sparsity.png")

# ---- A2. Lorenz curves --------------------------------------------------- #
fig = styled_fig()
header(fig, "Appendix · A2 — Concentration (Lorenz / Gini)",
       "Attention is concentrated; reader activity is near-uniform",
       "Distance from the diagonal = inequality. Books are highly concentrated; the flat "
       "reader curve is a fingerprint of the filtered ≥100-rating core.")
ax = fig.add_axes([0.1, 0.13, 0.8, 0.6])
style_ax(ax)
px, py = lorenz_points(book_pop.values)
ax.plot(px * 100, py * 100, color=TEAL, lw=2.6,
        label=f"Book popularity  (Gini {gini_pop:.2f})")
px2, py2 = lorenz_points(user_summary["n_ratings"].values)
ax.plot(px2 * 100, py2 * 100, color=TERRA, lw=2.6,
        label=f"Reader activity  (Gini {gini_user:.2f})")
ax.plot([0, 100], [0, 100], color=MUTED, ls="--", lw=1.3, label="Perfect equality")
ax.set_xlabel("Cumulative share of population (%)")
ax.set_ylabel("Cumulative share of ratings (%)")
ax.set_xlim(0, 100); ax.set_ylim(0, 100)
ax.legend(frameon=False, loc="upper left", fontsize=12)
footer(fig)
save(fig, "appendix_02_lorenz_gini.png")

# ---- A3. Zipf / power-law ------------------------------------------------ #
fig = styled_fig()
header(fig, "Appendix · A3 — Long-tail shape (Zipf fit)",
       f"Popularity follows a near power law (slope {zipf_slope:.2f})",
       "Rank-frequency on log-log axes; a straight line indicates power-law / Zipfian decay.")
ax = fig.add_axes([0.1, 0.13, 0.8, 0.6])
style_ax(ax)
ax.loglog(ranks, counts_sorted, color=TEAL, lw=2, label="Observed")
fit = 10 ** (zipf_intercept + zipf_slope * logx)
ax.loglog(ranks, fit, color=TERRA, ls="--", lw=2,
          label=f"Power-law fit (slope {zipf_slope:.2f}, R² {zipf_r2:.2f})")
ax.set_xlabel("Book popularity rank (log)")
ax.set_ylabel("Number of ratings (log)")
ax.grid(True, which="both", axis="both")
ax.legend(frameon=False, fontsize=12)
footer(fig)
save(fig, "appendix_03_zipf.png")

# ---- A4. Bias decomposition ---------------------------------------------- #
fig = styled_fig()
header(fig, "Appendix · A4 — Rating-bias decomposition",
       f"A simple bias model explains {r2_useritem*100:.0f}% of rating variance",
       "r ≈ μ + b_user + b_item. Why “predict the adjusted mean” is a strong RMSE baseline.")
ax1 = fig.add_axes([0.07, 0.13, 0.40, 0.58])
style_ax(ax1)
ax1.hist(b_user.values, bins=40, color=TEAL, edgecolor="white", alpha=0.85,
         label="reader bias b_user")
ax1.hist(b_item.values, bins=40, color=TERRA, edgecolor="white", alpha=0.6,
         label="book bias b_item")
ax1.axvline(0, color=INK, lw=1)
ax1.set_xlabel("Bias (stars vs global mean)")
ax1.set_ylabel("Count")
ax1.legend(frameon=False, fontsize=11)
ax2 = fig.add_axes([0.57, 0.13, 0.36, 0.58])
style_ax(ax2)
labels = ["Global\nmean", "+ reader\nbias", "+ book\nbias"]
rmses = [rmse_global, rmse_user, rmse_useritem]
bars = ax2.bar(labels, rmses, color=[MUTED, TEAL, TERRA], edgecolor="white", width=0.6)
for b, v in zip(bars, rmses):
    ax2.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
             ha="center", va="bottom", fontsize=12, fontweight="bold", color=INK)
ax2.set_ylabel("In-sample RMSE")
ax2.set_ylim(0, max(rmses) * 1.18)
footer(fig, "In-sample / illustrative — not a held-out score")
save(fig, "appendix_04_bias_decomposition.png")

# ---- A5. Empirical-Bayes shrinkage --------------------------------------- #
fig = styled_fig()
header(fig, "Appendix · A5 — Bayesian shrinkage of book averages",
       "Low-support books regress toward the mean",
       f"Raw average vs support, with the empirical-Bayes estimate (prior C={C:.0f} ratings).")
ax = fig.add_axes([0.09, 0.13, 0.84, 0.6])
style_ax(ax)
bs = book_summary.dropna(subset=["observed_avg"])
# jitter x a touch so the discrete integer supports don't overplot into bars
jitter = np.exp(np.linspace(-0.04, 0.04, len(bs)))
ax.scatter(bs["observed_ratings"] * jitter, bs["observed_avg"], s=14, alpha=0.18,
           color=TEAL, label="raw observed average", edgecolors="none")
ax.scatter(bs["observed_ratings"] * jitter, bs["shrunk_avg"], s=14, alpha=0.45,
           color=TERRA, label="empirical-Bayes (shrunk)", edgecolors="none")
ax.axhline(mu, color=INK, ls="--", lw=1.3, label=f"global mean {mu:.2f}")
ax.set_xscale("log")
ax.set_xlabel("Ratings per book (log scale = support)")
ax.set_ylabel("Average rating")
ax.set_ylim(0.8, 5.2)
ax.annotate("low support →\nshrunk to the mean", xy=(1.5, mu), xytext=(1.15, 1.6),
            color=INK, fontsize=11, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
leg = ax.legend(frameon=False, fontsize=11, loc="upper right")
for lh in leg.legend_handles:
    lh.set_alpha(1)
footer(fig, f"Raw vs shrunk top-10 overlap: "
            f"{stats['shrinkage']['top10_overlap_raw_vs_shrunk']}/10")
save(fig, "appendix_05_shrinkage.png")

# ---- A6. Engagement segmentation ----------------------------------------- #
fig = styled_fig()
header(fig, "Appendix · A6 — Reader engagement segmentation",
       f"Contribution is unusually even — top 10% supply only {segmentation['Top 10%']:.0f}%",
       "No “whales vs minnows” here (organic data would be far steeper) — confirming a "
       "deliberately balanced reader sample, not production traffic.")
ax = fig.add_axes([0.1, 0.14, 0.82, 0.58])
style_ax(ax)
seg_labels = list(segmentation.keys())
seg_vals = list(segmentation.values())
bars = ax.barh(seg_labels[::-1], seg_vals[::-1], color=TEAL, edgecolor="white")
bars[-1].set_color(TERRA)
for b, v in zip(bars, seg_vals[::-1]):
    ax.text(v + 1, b.get_y() + b.get_height() / 2, f"{v:.0f}%",
            va="center", fontsize=12, fontweight="bold", color=INK)
ax.set_xlabel("Share of all ratings contributed (%)")
ax.set_xlim(0, 100)
ax.grid(axis="y", visible=False)
footer(fig)
save(fig, "appendix_06_engagement.png")

# ---- A7. Catalog vintage ------------------------------------------------- #
fig = styled_fig()
header(fig, "Appendix · A7 — Catalog vintage",
       f"Median publication year {stats['vintage']['median_year']:.0f} · "
       f"{stats['vintage']['pct_since_2000']:.0f}% published since 2000",
       "Publication-year distribution (years >0; pre-1900 clipped for readability).")
ax = fig.add_axes([0.09, 0.14, 0.84, 0.58])
style_ax(ax)
yclip = yr.clip(lower=1900)
ax.hist(yclip, bins=np.arange(1900, 2027, 3), color=TEAL, edgecolor="white")
ax.axvline(yr.median(), color=TERRA, ls="--", lw=1.8,
           label=f"median {yr.median():.0f}")
ax.set_xlabel("Original publication year (clipped at 1900)")
ax.set_ylabel("Number of books")
ax.yaxis.set_major_formatter(thousands)
ax.legend(frameon=False, fontsize=12)
footer(fig, f"{stats['vintage']['anomalies_year_lt0_or_gt2026']} year anomalies (<0 or >2026)")
save(fig, "appendix_07_vintage.png")

# ---- A8. Metadata completeness ------------------------------------------- #
fig = styled_fig()
header(fig, "Appendix · A8 — Metadata completeness",
       "Side-information gaps constrain a content / hybrid model",
       "Missingness reflects the post-clean state models see (loader fills authors/year).")
ax = fig.add_axes([0.1, 0.14, 0.82, 0.58])
style_ax(ax)
mlabels = list(stats["metadata_missing_pct"].keys())
mvals = [stats["metadata_missing_pct"][k] or 0 for k in mlabels]
bars = ax.barh(mlabels[::-1], mvals[::-1],
               color=[TERRA if v > 1 else TEAL for v in mvals[::-1]],
               edgecolor="white")
for b, v in zip(bars, mvals[::-1]):
    ax.text(v + 0.3, b.get_y() + b.get_height() / 2, f"{v:.1f}%",
            va="center", fontsize=12, fontweight="bold", color=INK)
ax.set_xlabel("Missing (%)")
ax.set_xlim(0, max(max(mvals) * 1.25, 5))
ax.grid(axis="y", visible=False)
footer(fig, "language_code / isbn gaps are real; authors / year filled by loader")
save(fig, "appendix_08_metadata.png")

# ---- A9. Data-quality table ---------------------------------------------- #
fig = styled_fig()
header(fig, "Appendix · A9 — Data-quality audit",
       "The table is clean after the loader's contract enforcement",
       "Sanity checks that license trust in every downstream metric.")
ax = fig.add_axes([0.08, 0.1, 0.84, 0.62])
ax.axis("off")
tbl = ax.table(cellText=[[c, f"{n:,}"] for c, n in
                         zip(quality["check"], quality["count"])],
               colLabels=["Check", "Count"],
               cellLoc="left", colLoc="left", loc="center",
               colWidths=[0.7, 0.3])
tbl.auto_set_font_size(False)
tbl.set_fontsize(13)
tbl.scale(1, 2.0)
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor(GRID)
    if r == 0:
        cell.set_facecolor(TEAL)
        cell.set_text_props(color="white", fontweight="bold")
    else:
        cell.set_facecolor(PANEL if r % 2 else "#F4EFE6")
footer(fig)
save(fig, "appendix_09_data_quality.png")

# =========================================================================== #
# WRITE stats.json + markdown report
# =========================================================================== #
with open(os.path.join(REPORTS_DIR, "eda_stats.json"), "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=2)
print("  saved eda_stats.json")

ov = stats["overview"]
md = f"""# Project 2 — Book Recommender EDA

*Generated by `reports/run_eda.py` from `Books.csv` + `Ratings.csv` via `bookrec.data_loader`.
All figures are 16:9, 200 dpi, slide-ready (`reports/figures/`).*

---

## Executive summary

The data has four structural features that should drive every modeling and
product decision:

1. **It is extremely sparse.** {ov['n_users']:,} readers × {ov['n_books_rated']:,} books
   is a matrix that is **{ov['sparsity_pct']:.2f}% empty** (density {ov['density_pct']:.3f}%).
   Collaborative filtering must work from very little overlap.
2. **It is blockbuster-driven.** The top 1% of books capture
   **{top_pct_share['top_1pct_books']:.0f}%** of all ratings
   (Gini {gini_pop:.2f}). Recommending on popularity alone is easy but collapses
   catalog coverage — so we report coverage next to accuracy.
3. **Feedback is positively skewed.** **{ov['pct_4_or_5_star']:.0f}%** of ratings are
   4-5★ and the mean is {ov['global_mean_rating']:.2f}/5. RMSE rewards predicting
   "high," which makes a popularity / bias baseline a genuinely strong competitor.
4. **This is a filtered "dense-reader core," and cold-start is an *item* problem.**
   Every reader carries **{stats['sample_design']['min_ratings_per_user']}–{stats['sample_design']['max_ratings_per_user']} ratings**
   (median {stats['sample_design']['median_ratings_per_user']:.0f}), so user-side cold-start
   is ~0% *by construction*. The thin-signal risk sits on the catalog:
   **{stats['sample_design']['pct_books_lt_10']:.0f}%** of rated books have <10 ratings and
   **{ov['n_books_never_rated']:,}** books are never rated at all.

> **Generalization caveat (read before trusting any score):** because readers were
> pre-filtered to ≥100 ratings, accuracy here is *optimistic* relative to production,
> where most users are sparse. Validate against held-out users and stress-test on
> low-history cohorts before shipping.

The recommended architecture is the one in notebook 01: a strong bias/popularity
baseline, CF (UBCF/IBCF/SVD) where signal is sufficient, Bayesian-shrunk averages so
thin-support books don't dominate, and a popularity fallback for cold items/users.

### Executive figures
| Figure | File |
|---|---|
| Data at a glance (KPIs) | `figures/exec_01_kpi_scorecard.png` |
| The discovery problem (long tail) | `figures/exec_02_long_tail.png` |
| Ratings skew positive | `figures/exec_03_rating_skew.png` |
| Cold-start majority | `figures/exec_04_cold_start.png` |

---

## Headline numbers

| Metric | Value |
|---|---|
| Ratings | {ov['n_ratings']:,} |
| Unique readers | {ov['n_users']:,} |
| Books rated (of {ov['n_catalog']:,} in catalog) | {ov['n_books_rated']:,} |
| Books never rated in sample | {ov['n_books_never_rated']:,} |
| Matrix density | {ov['density_pct']:.3f}% |
| Mean / median rating | {ov['global_mean_rating']:.2f} / {ov['median_rating']:.0f} |
| Share 4-5★ | {ov['pct_4_or_5_star']:.0f}% |
| Avg ratings per reader / book | {ov['avg_ratings_per_user']:.1f} / {ov['avg_ratings_per_book']:.1f} |
| Ratings per reader (min / median / max) | {stats['sample_design']['min_ratings_per_user']} / {stats['sample_design']['median_ratings_per_user']:.0f} / {stats['sample_design']['max_ratings_per_user']} |
| Books with <10 ratings (item cold-start) | {stats['cold_start_books']['books_with_lt_10']:,} ({stats['sample_design']['pct_books_lt_10']:.0f}%) |
| Books never rated in sample | {ov['n_books_never_rated']:,} |
| Gini — book popularity | {gini_pop:.2f} |
| Gini — reader activity | {gini_user:.2f} |
| Zipf log-log slope (R²) | {zipf_slope:.2f} ({zipf_r2:.2f}) |
| Bias model R² (μ + b_user + b_item) | {r2_useritem:.2f} |
| RMSE: global mean → +user → +item | {rmse_global:.3f} → {rmse_user:.3f} → {rmse_useritem:.3f} |
| Top 10% of readers' share of ratings | {segmentation['Top 10%']:.0f}% |
| Observed vs Goodreads avg (Pearson r) | {stats['calibration_observed_vs_goodreads_pearson_r']} |

---

## Technical appendix (figures)

| # | Topic | What it shows | File |
|---|---|---|---|
| A1 | Matrix sparsity | Even the densest 200×200 corner is mostly empty | `figures/appendix_01_sparsity.png` |
| A2 | Lorenz / Gini | Inequality of attention (books) and contribution (readers) | `figures/appendix_02_lorenz_gini.png` |
| A3 | Zipf fit | Long tail is near power-law (slope {zipf_slope:.2f}) | `figures/appendix_03_zipf.png` |
| A4 | Bias decomposition | μ + b_user + b_item explains {r2_useritem*100:.0f}% of variance | `figures/appendix_04_bias_decomposition.png` |
| A5 | Bayesian shrinkage | Low-support averages regress to the mean | `figures/appendix_05_shrinkage.png` |
| A6 | Engagement segments | Whales vs minnows contribution curve | `figures/appendix_06_engagement.png` |
| A7 | Catalog vintage | Publication-year distribution | `figures/appendix_07_vintage.png` |
| A8 | Metadata completeness | Side-info gaps for content/hybrid models | `figures/appendix_08_metadata.png` |
| A9 | Data-quality audit | Clean table after loader contract | `figures/appendix_09_data_quality.png` |

### Modeling implications
- **Baseline first.** The additive bias model already reaches RMSE
  {rmse_useritem:.3f} (from {rmse_global:.3f} at the global mean) — any CF/SVD model
  must beat this to justify its complexity.
- **Report coverage + ranking metrics**, not RMSE alone — accuracy is inflated by the
  4-5★ skew, and popularity concentration hides poor catalog coverage.
- **Plan for cold start on the item side** — {stats['sample_design']['pct_books_lt_10']:.0f}%
  of books have <10 ratings and {ov['n_books_never_rated']:,} have none; use Bayesian-shrunk
  averages so thin-support books don't dominate leaderboards, and a content fallback for
  unrated titles. User cold-start is hidden by the ≥100-rating filter, so validate on
  held-out / low-history users before trusting offline accuracy.
- **Content/hybrid is constrained** by metadata gaps (`language_code`, `isbn`), which is
  why the content path falls back to title + authors.

*Raw numbers: `eda_stats.json` · tables: `eda_tables/*.csv`.*
"""
with open(os.path.join(REPORTS_DIR, "eda_summary.md"), "w", encoding="utf-8") as f:
    f.write(md)
print("  saved eda_summary.md")

# Console digest -----------------------------------------------------------
print("\n" + "=" * 64)
print("EDA COMPLETE — headline numbers")
print("=" * 64)
print(f"ratings={n_ratings:,}  readers={n_users:,}  books_rated={n_books_rated:,}"
      f"  catalog={n_catalog:,}")
print(f"matrix density        : {density*100:.3f}%  (sparsity {(1-density)*100:.2f}%)")
print(f"mean rating           : {mu:.2f}   (4-5 star share {ov['pct_4_or_5_star']:.0f}%)")
print(f"cold-start readers <5 : {stats['cold_start_users']['users_with_lt_5']:,}"
      f" ({stats['cold_start_users']['pct_lt_5']:.0f}%)")
print(f"top 1% books share    : {top_pct_share['top_1pct_books']:.0f}% of ratings")
print(f"Gini popularity/users : {gini_pop:.2f} / {gini_user:.2f}")
print(f"Zipf slope (R2)       : {zipf_slope:.2f} ({zipf_r2:.2f})")
print(f"bias model R2         : {r2_useritem:.2f}")
print(f"RMSE mean->u->u+i     : {rmse_global:.3f} -> {rmse_user:.3f} -> {rmse_useritem:.3f}")
print(f"top-10% readers share : {segmentation['Top 10%']:.0f}% of ratings")
print(f"\n{len(_saved)} figures + report written to {REPORTS_DIR}")
