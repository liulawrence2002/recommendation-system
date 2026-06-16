"""
Collaborative filtering + a popularity baseline.

Same `surprise` building blocks as the Week-3 class notebook (KNNBasic UBCF/IBCF,
BaselineOnly), plus SVD, plus a non-personalized popularity/mean baseline. Every
model exposes a uniform .predict(user, book) and .predict_for_user(user, ids).

`surprise` is imported lazily so the popularity baseline (and the content/LLM
paths) still run where the compiled scikit-surprise wheel isn't installed.
"""
from __future__ import annotations

import pandas as pd

RATING_SCALE = (1.0, 5.0)   # this dataset uses 1..5 stars


def _require_surprise():
    # Keep Surprise optional at import time: the app can still run its
    # popularity baseline and LLM fallback on machines without compiled wheels.
    try:
        import surprise  # noqa: F401
        return surprise
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "scikit-surprise is needed for collaborative filtering.\n"
            "  pip install 'numpy<2.0' && pip install scikit-surprise\n"
            "  (or: conda install -c conda-forge scikit-surprise)\n"
            f"original error: {e}"
        )


def build_dataset(ratings: pd.DataFrame):
    """Wrap a [user_id, book_id, rating] frame as a Surprise Dataset."""
    surprise = _require_surprise()
    reader = surprise.Reader(rating_scale=RATING_SCALE)
    return surprise.Dataset.load_from_df(
        ratings[["user_id", "book_id", "rating"]], reader)


def make_model(kind: str = "svd", k: int = 10, sim_name: str | None = None):
    """Factory: 'svd', 'ubcf' (user-based), 'ibcf' (item-based),
    'baseline' (global + user + item means).

    sim_name overrides the KNN similarity for ubcf/ibcf (e.g. "pearson_baseline").
    When None the historical defaults apply: ubcf -> pearson, ibcf -> cosine.
    The corrected Top-N audit picks ibcf with "pearson_baseline" (see app.py).
    """
    surprise = _require_surprise()
    if kind == "svd":
        return surprise.SVD(n_factors=50, n_epochs=20, random_state=6604)
    if kind == "ubcf":
        sim = {"name": sim_name or "pearson", "user_based": True}
        return surprise.KNNBasic(k=k, sim_options=sim, verbose=False)
    if kind == "ibcf":
        sim = {"name": sim_name or "cosine", "user_based": False}
        return surprise.KNNBasic(k=k, sim_options=sim, verbose=False)
    if kind == "baseline":
        return surprise.BaselineOnly(verbose=False)
    raise ValueError(f"unknown kind: {kind!r}")


class PopularityModel:
    """Non-personalized popularity/mean baseline - the benchmark CF must beat.

    Pure Python (no surprise), so it always runs. For RMSE it predicts each
    book's mean rating, shrunk toward the global mean for thinly-rated books;
    for Top-N it ranks by that popularity score.
    """

    def __init__(self, shrinkage: int = 10):
        self.shrinkage = shrinkage
        self.global_mean = 3.5
        self.book_mean = {}

    def fit(self, ratings: pd.DataFrame):
        self.global_mean = float(ratings["rating"].mean())
        g = ratings.groupby("book_id")["rating"]
        means, counts = g.mean(), g.count()
        # Bayesian-shrunk mean: pull low-count books toward the global mean
        # so one or two enthusiastic ratings do not dominate the baseline.
        shrunk = (counts * means + self.shrinkage * self.global_mean) / (counts + self.shrinkage)
        self.book_mean = shrunk.to_dict()
        return self

    def predict(self, user_id, book_id) -> float:
        return self.book_mean.get(book_id, self.global_mean)

    def predict_for_user(self, user_id, book_ids) -> pd.Series:
        est = [self.predict(user_id, b) for b in book_ids]
        return pd.Series(est, index=list(book_ids), name="pop_score")


class CFModel:
    """Uniform wrapper: train, then .predict(user, book) -> estimated rating."""

    def __init__(self, kind: str = "svd", k: int = 10, sim_name: str | None = None):
        self.kind = kind
        self.k = k
        self.sim_name = sim_name
        self.algo = make_model(kind, k, sim_name)
        self._global_mean = 3.5

    def fit(self, ratings: pd.DataFrame, trainset=None):
        if trainset is None:
            trainset = build_dataset(ratings).build_full_trainset()
        # Accepting an external trainset lets evaluation notebooks reuse the
        # exact same split instead of silently rebuilding on all ratings.
        self.algo.fit(trainset)
        self._global_mean = trainset.global_mean
        return self

    def predict(self, user_id, book_id) -> float:
        return self.algo.predict(user_id, book_id).est

    def predict_for_user(self, user_id, book_ids) -> pd.Series:
        est = [self.predict(user_id, b) for b in book_ids]
        return pd.Series(est, index=list(book_ids), name="cf_score")
