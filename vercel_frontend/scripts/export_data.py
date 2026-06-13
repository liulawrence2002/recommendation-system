"""
One-time data export: bake the BookRec framework into static JSON for Vercel.

The Streamlit app trains collaborative-filtering models live with scikit-surprise.
That stack (compiled wheels, NumPy<2, multi-second cold-start training) does not
belong in a Vercel serverless function. Instead we run the *same* recommendation
logic once here, offline, and emit small JSON files the Next.js app reads:

    public/data/books.json   - the catalog (id, title, authors, year, covers, ...)
    public/data/scores.json  - per-book CF score for the auto-reader, per model,
                               aligned to the books.json order (popularity,
                               baseline, ubcf, ibcf, svd) + the reader's seen ids
    public/data/stats.json   - EDA aggregates, headline metrics, the model-audit
                               table, and the author/decade filter options

The candidate generation in app.py (recommend.recommend_top_n) is just:
  popularity floor -> exclude seen -> optional author/decade filter -> sort by the
  model's score -> Top-N.  With the per-book scores baked here, the browser
  reproduces that exactly and interactively (change model / min-ratings / filters
  and the Top-N updates) without any Python at request time.

Run it with the system interpreter that has pandas/scipy/sklearn:

    cd vercel_frontend
    python scripts/export_data.py

It imports the project's real data_loader so the catalog (encoding fixes, the
column contract) is identical to the app's.
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# --- locate the repo so we can reuse bookrec/src + bookrec/data ----------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_FRONTEND_DIR = os.path.dirname(_SCRIPT_DIR)
_REPO_ROOT = os.path.dirname(_FRONTEND_DIR)
_BOOKREC_DIR = os.path.join(_REPO_ROOT, "bookrec")
_DATA_DIR = os.path.join(_BOOKREC_DIR, "data")
_OUT_DIR = os.path.join(_FRONTEND_DIR, "public", "data")

sys.path.insert(0, _BOOKREC_DIR)
from src import data_loader  # noqa: E402  (reuse the app's exact loader)

# Constants mirrored from app.py so the export matches the app's defaults.
DEFAULT_UBCF_K = 21
SHRINKAGE = 10           # PopularityModel Bayesian shrinkage
BASELINE_REG = 10.0      # regularization for the baseline bias terms
SVD_FACTORS = 50
MIN_DECADE_BOOKS = 50    # app.py decade_options binning threshold

# The canonical offline hold-out result the app reports (NOTEBOOK_METRICS).
NOTEBOOK_METRICS = [
    ("Baseline", 0.6568, 0.7912, 0.7178),
    ("UBCF (pearson)", 0.6586, 0.7930, 0.7196),
    ("IBCF (cosine)", 0.6414, 0.7734, 0.7012),
]


def log(msg: str) -> None:
    print(f"[export] {msg}", flush=True)


def goodreads_url(cover_url: str, title: str, authors: str) -> str:
    """Mirror of app.py `_goodreads_url`: the cover filename embeds the goodreads
    book id, which gives a direct page; otherwise fall back to a title+author
    search so the link always resolves."""
    m = re.search(r"/(\d+)\.[a-zA-Z]+$", cover_url or "")
    if m:
        return f"https://www.goodreads.com/book/show/{m.group(1)}"
    import urllib.parse
    q = urllib.parse.quote_plus(f"{title} {authors}".strip())
    return f"https://www.goodreads.com/search?q={q}"


def safe_int(value, default: int = 0) -> int:
    try:
        if value != value:  # NaN
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def safe_float(value, default: float = 0.0, ndigits: int = 4) -> float:
    try:
        if value != value:
            return default
        return round(float(value), ndigits)
    except (TypeError, ValueError):
        return default


# --- CF scorers (numpy/scipy reimplementations of the surprise models) ---------
# Each returns a 1-D array of predicted ratings for the auto-reader, indexed by
# the item column order (col_of[book_id]).

def popularity_scores(book_means: dict, global_mean: float, book_ids) -> dict:
    """cf_model.PopularityModel: Bayesian-shrunk per-book mean, global-mean prior."""
    return {b: book_means.get(b, global_mean) for b in book_ids}


def baseline_scores(ratings: pd.DataFrame, auto_uid, global_mean: float) -> dict:
    """surprise BaselineOnly-style: mu + b_user + b_item with regularized means.

    b_i = sum(r_ui - mu) / (reg + n_i);  b_u = sum(r_ui - mu - b_i) / (reg + n_u).
    Returns {book_id: mu + b_auto + b_i} for every rated book (others get mu).
    """
    dev = ratings["rating"] - global_mean
    bi = dev.groupby(ratings["book_id"]).sum() / (
        BASELINE_REG + ratings.groupby("book_id")["rating"].count()
    )
    bi_map = bi.to_dict()
    a = ratings[ratings["user_id"] == auto_uid]
    b_auto = float(
        (a["rating"] - global_mean - a["book_id"].map(bi_map).fillna(0.0)).sum()
        / (BASELINE_REG + len(a))
    )
    return {b: global_mean + b_auto + bi_map.get(b, 0.0) for b in bi_map}, b_auto


def ubcf_scores(R_csr, R_csc, a_row: int, user_means: np.ndarray, k: int):
    """KNNBasic(pearson, user_based=True): per item, weighted avg of the k nearest
    neighbours (by similarity) who rated it. Similarity is mean-centered cosine
    (== Pearson over the shared support), the standard UBCF kernel."""
    # Mean-center observed entries, then cosine over users == Pearson-style sim.
    Rc = R_csr.copy().tolil()
    coo = R_csr.tocoo()
    centered = R_csr.copy().astype(np.float64)
    centered.data = centered.data - user_means[coo.row]
    centered = centered.tocsr()
    sim = cosine_similarity(centered[a_row], centered).ravel()
    sim[a_row] = 0.0  # never neighbour-of-self

    n_items = R_csc.shape[1]
    out = np.full(n_items, np.nan)
    indptr, indices, data = R_csc.indptr, R_csc.indices, R_csc.data
    for j in range(n_items):
        lo, hi = indptr[j], indptr[j + 1]
        if lo == hi:
            continue
        raters = indices[lo:hi]
        s = sim[raters]
        pos = s > 0
        if not pos.any():
            continue
        raters, s, r = raters, s[pos], data[lo:hi][pos]
        raters = raters[pos]
        if s.shape[0] > k:
            top = np.argpartition(s, -k)[-k:]
            s, r = s[top], r[top]
        denom = np.abs(s).sum()
        if denom > 0:
            out[j] = float(np.dot(s, r) / denom)
    return out


def ibcf_scores(R_csc, auto_items: np.ndarray, auto_ratings: np.ndarray, k: int):
    """KNNBasic(cosine, item_based=True): per candidate item, weighted avg of the
    user's ratings on the k most-similar items they have rated."""
    n_items = R_csc.shape[1]
    item_user = R_csc.T.tocsr()  # items x users
    rated_mat = item_user[auto_items]  # (n_rated x users)
    sims = cosine_similarity(item_user, rated_mat)  # (n_items x n_rated)
    out = np.full(n_items, np.nan)
    for j in range(n_items):
        s = sims[j].copy()
        s[s <= 0] = 0.0
        if not s.any():
            continue
        if s.shape[0] > k:
            keep = np.argpartition(s, -k)[-k:]
            mask = np.zeros_like(s, dtype=bool)
            mask[keep] = True
            s = np.where(mask, s, 0.0)
        denom = s.sum()
        if denom > 0:
            out[j] = float(np.dot(s, auto_ratings) / denom)
    return out


def svd_scores(R_csr, a_row: int, global_mean: float):
    """Truncated-SVD matrix factorization, reconstructing the auto-reader's row.
    A light stand-in for surprise.SVD; centered by the global mean so the scores
    sit in the rating range."""
    centered = R_csr.astype(np.float64).copy()
    centered.data = centered.data - global_mean
    svd = TruncatedSVD(n_components=SVD_FACTORS, random_state=6604)
    U = svd.fit_transform(centered)          # users x factors
    recon = U[a_row] @ svd.components_        # 1 x items
    return recon + global_mean


# --- filter options (mirror app.py author_options / decade_options) ------------

def author_options(books: pd.DataFrame) -> list:
    seen = set()
    for cell in books["authors"].dropna():
        for part in str(cell).split(","):
            v = part.strip()
            if v and v != "Unknown":
                seen.add(v)
    return sorted(seen)


def decade_options(books: pd.DataFrame) -> list:
    years = pd.to_numeric(books["original_publication_year"], errors="coerce").dropna()
    years = years[years > 0]
    if years.empty:
        return []
    counts = (years // 10 * 10).astype(int).value_counts()
    decades = sorted(counts.index)
    dense = [d for d in decades if counts[d] >= MIN_DECADE_BOOKS]
    if len(dense) < 2 or dense[0] == decades[0]:
        return [f"{d}s" for d in decades]
    cutoff = dense[0]
    return [f"Before {cutoff}s"] + [f"{d}s" for d in decades if d >= cutoff]


def main() -> None:
    os.makedirs(_OUT_DIR, exist_ok=True)
    log(f"loading catalog from {_DATA_DIR}")
    ratings, books = data_loader.load(_DATA_DIR)
    books = books.reset_index(drop=True)
    log(f"ratings={ratings.shape} books={books.shape}")

    auto_uid = ratings["user_id"].value_counts().idxmax()
    global_mean = float(ratings["rating"].mean())
    log(f"auto-reader uid={auto_uid} "
        f"({int(ratings['user_id'].value_counts().max())} ratings)")

    # dataset rating counts per book (popularity floor + min-ratings filter)
    dataset_counts = ratings.groupby("book_id").size()
    dataset_counts_map = dataset_counts.to_dict()

    # ---- books.json ----
    book_ids = books["book_id"].tolist()
    books_out = []
    for row in books.itertuples(index=False):
        cover = "" if pd.isna(getattr(row, "small_image_url", "")) else str(getattr(row, "small_image_url", ""))
        image = "" if pd.isna(getattr(row, "image_url", "")) else str(getattr(row, "image_url", ""))
        title = str(getattr(row, "title", "") or "Untitled")
        authors = str(getattr(row, "authors", "") or "Unknown")
        cover_for_link = cover or image
        books_out.append({
            "id": safe_int(row.book_id),
            "title": title,
            "authors": authors,
            "year": safe_int(getattr(row, "original_publication_year", 0)),
            "avgRating": safe_float(getattr(row, "average_rating", 0.0), ndigits=2),
            "goodreadsCount": safe_int(getattr(row, "ratings_count", 0)),
            "datasetCount": safe_int(dataset_counts_map.get(row.book_id, 0)),
            "cover": cover,
            "image": image,
            "url": goodreads_url(cover_for_link, title, authors),
        })
    log(f"books.json -> {len(books_out)} books")

    # ---- per-model scores for the auto-reader ----
    # sparse user x item matrix
    uids = ratings["user_id"].unique()
    row_of = {u: i for i, u in enumerate(uids)}
    col_of = {b: i for i, b in enumerate(book_ids)}
    rows = ratings["user_id"].map(row_of).to_numpy()
    cols = ratings["book_id"].map(col_of).to_numpy()
    vals = ratings["rating"].to_numpy(dtype=np.float64)
    n_users, n_items = len(uids), len(book_ids)
    R_csr = sparse.csr_matrix((vals, (rows, cols)), shape=(n_users, n_items))
    R_csc = R_csr.tocsc()
    a_row = row_of[auto_uid]

    # per-user mean over observed entries (for the UBCF kernel)
    user_sums = np.asarray(R_csr.sum(axis=1)).ravel()
    user_nnz = np.diff(R_csr.indptr)
    user_means = np.divide(user_sums, np.maximum(user_nnz, 1))

    # popularity (Bayesian-shrunk mean)
    g = ratings.groupby("book_id")["rating"]
    means, counts = g.mean(), g.count()
    shrunk = (counts * means + SHRINKAGE * global_mean) / (counts + SHRINKAGE)
    book_means = shrunk.to_dict()
    log("scoring: popularity")
    pop = popularity_scores(book_means, global_mean, book_ids)

    log("scoring: baseline")
    base_map, b_auto = baseline_scores(ratings, auto_uid, global_mean)

    log("scoring: ubcf (pearson, user-based)")
    ubcf = ubcf_scores(R_csr, R_csc, a_row, user_means, DEFAULT_UBCF_K)

    log("scoring: ibcf (cosine, item-based)")
    auto_col_block = R_csr[a_row].tocoo()
    auto_items = auto_col_block.col
    auto_ratings = auto_col_block.data
    ibcf = ibcf_scores(R_csc, auto_items, auto_ratings, DEFAULT_UBCF_K)

    log("scoring: svd")
    svd = svd_scores(R_csr, a_row, global_mean)

    def clip(x: float) -> float:
        if x != x:  # NaN
            return round(global_mean, 4)
        return round(float(min(5.0, max(1.0, x))), 4)

    auto_user_mean = float(user_means[a_row])
    models = {"popularity": [], "baseline": [], "ubcf": [], "ibcf": [], "svd": []}
    for idx, b in enumerate(book_ids):
        models["popularity"].append(clip(pop.get(b, global_mean)))
        models["baseline"].append(clip(base_map.get(b, global_mean + b_auto)))
        models["ubcf"].append(clip(ubcf[idx] if ubcf[idx] == ubcf[idx] else auto_user_mean))
        models["ibcf"].append(clip(ibcf[idx] if ibcf[idx] == ibcf[idx] else auto_user_mean))
        models["svd"].append(clip(svd[idx]))

    seen_ids = sorted(int(b) for b in ratings.loc[ratings["user_id"] == auto_uid, "book_id"])
    scores_out = {
        "autoUser": safe_int(auto_uid),
        "autoUserRatings": int(len(seen_ids)),
        "seen": seen_ids,
        "models": models,
    }
    log(f"scores.json -> {len(models)} models x {n_items} books")

    # ---- EDA / stats (mirror app.py dataset_insights) ----
    rating_dist = ratings["rating"].round(1).value_counts().sort_index()
    per_user = ratings.groupby("user_id").size()
    per_book = ratings.groupby("book_id").size().sort_values(ascending=False)
    top10pct_n = max(1, int(len(per_book) * 0.10))

    top = (
        per_book.head(10).rename("ratings").reset_index()
        .merge(books[["book_id", "title", "authors", "average_rating"]], on="book_id", how="left")
    )
    top_books = [
        {
            "title": str(r.title),
            "authors": str(r.authors),
            "ratings": safe_int(r.ratings),
            "avgRating": safe_float(r.average_rating, ndigits=2),
        }
        for r in top.itertuples(index=False)
    ]

    years = pd.to_numeric(books["original_publication_year"], errors="coerce")
    years = years[years > 0]
    decade_counts = (years // 10 * 10).astype(int).value_counts().sort_index()

    n_users_total = int(ratings["user_id"].nunique())
    n_books_total = int(len(books))
    n_ratings_total = int(len(ratings))
    sparsity = 1 - n_ratings_total / (n_users_total * n_books_total)

    stats_out = {
        "headline": {
            "users": n_users_total,
            "books": n_books_total,
            "ratings": n_ratings_total,
            "sparsity": round(sparsity, 5),
        },
        "eda": {
            "meanRating": round(float(ratings["rating"].mean()), 3),
            "pct4plus": round(float((ratings["rating"] >= 4).mean()), 3),
            "perUserMedian": int(per_user.median()),
            "perUserMean": round(float(per_user.mean()), 1),
            "perUserMax": int(per_user.max()),
            "perBookMedian": int(per_book.median()),
            "top10pctShare": round(float(per_book.head(top10pct_n).sum() / per_book.sum()), 3),
            "sparsity": round(sparsity, 5),
            "ratingDist": [
                {"rating": float(k), "count": int(v)} for k, v in rating_dist.items()
            ],
            "decadeDist": [
                {"decade": f"{int(k)}s", "count": int(v)} for k, v in decade_counts.items()
            ],
            "topBooks": top_books,
        },
        "audit": {
            "notebookMetrics": [
                {"model": m, "p10": p, "r10": r, "f1": f}
                for (m, p, r, f) in NOTEBOOK_METRICS
            ],
            "seed": 6604,
            "k": 10,
        },
        "filters": {
            "authors": author_options(books),
            "decades": decade_options(books),
        },
        "defaults": {
            "ubcfK": DEFAULT_UBCF_K,
            "minRatings": 20,
            "candidateCount": 10,
        },
    }
    log(f"stats.json -> {len(stats_out['filters']['authors'])} authors, "
        f"{len(stats_out['filters']['decades'])} decade bins")

    # ---- write ----
    def dump(name: str, payload) -> None:
        path = os.path.join(_OUT_DIR, name)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"))
        kb = os.path.getsize(path) / 1024
        log(f"wrote {name} ({kb:.0f} KB)")

    dump("books.json", books_out)
    dump("scores.json", scores_out)
    dump("stats.json", stats_out)
    log("done.")


if __name__ == "__main__":
    main()
