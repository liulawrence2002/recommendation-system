"""
Evaluation — the part most projects get wrong (guide Section 11).

Provides:
  * train_test_split_ratings : random hold-out (use a TIME split on real data!)
  * rmse                      : rating-accuracy for a CF model
  * ranking_metrics          : Precision@K, Recall@K, NDCG@K from a scorer
  * coverage                  : catalog coverage of the recommendations

The ranking metrics generalize the class notebook's precision_recall_at_k to
any scorer (CF, content, or hybrid) and add NDCG + coverage so you can show
that accuracy and ranking quality do not always agree.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from collections import defaultdict


def train_test_split_ratings(ratings: pd.DataFrame, test_size: float = 0.1,
                             seed: int = 6604, time_col: str | None = None):
    """Random split by default. If you have timestamps, pass time_col to get a
    proper TEMPORAL split (train on the past, test on the future) — strongly
    preferred for a batch recommender (guide Section 11.2)."""
    if time_col and time_col in ratings.columns:
        r = ratings.sort_values(time_col)
        cut = int(len(r) * (1 - test_size))
        return r.iloc[:cut].copy(), r.iloc[cut:].copy()
    test = ratings.sample(frac=test_size, random_state=seed)
    train = ratings.drop(test.index)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def rmse(cf_model, test: pd.DataFrame) -> float:
    """Root mean squared error of predicted vs. true ratings (CF model)."""
    err = [(cf_model.predict(r.user_id, r.book_id) - r.rating) ** 2
           for r in test.itertuples(index=False)]
    return float(np.sqrt(np.mean(err))) if err else float("nan")


def _dcg(relevances) -> float:
    return sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))


def ranking_metrics(score_user_items, train: pd.DataFrame, test: pd.DataFrame,
                    k: int = 10, threshold: float = 4.0):
    """Precision@K, Recall@K, NDCG@K averaged over users.

    score_user_items(user_id, book_ids) -> pd.Series of scores (higher=better).
    A test book is 'relevant' if its held-out true rating >= threshold.
    We rank each user's TEST items by the model's score (class-style).
    """
    test_by_user = defaultdict(list)
    for r in test.itertuples(index=False):
        test_by_user[r.user_id].append((r.book_id, r.rating))

    precisions, recalls, ndcgs = [], [], []
    for uid, items in test_by_user.items():
        # Evaluation ranks only the held-out items for that user. This mirrors
        # the class exercise and isolates ordering quality from candidate recall.
        book_ids = [b for b, _ in items]
        true_r = dict(items)
        scores = score_user_items(uid, book_ids)
        ranked = sorted(book_ids, key=lambda b: scores.get(b, 0.0), reverse=True)

        n_relevant = sum(1 for _, t in items if t >= threshold)
        topk = ranked[:k]
        n_hits = sum(1 for b in topk if true_r[b] >= threshold)

        precisions.append(n_hits / k)
        if n_relevant > 0:
            recalls.append(n_hits / n_relevant)
        # NDCG with binary relevance
        # Ideal DCG uses the user's own held-out relevance labels as the ceiling.
        gains = [1.0 if true_r[b] >= threshold else 0.0 for b in topk]
        ideal = sorted([1.0 if t >= threshold else 0.0 for _, t in items], reverse=True)[:k]
        idcg = _dcg(ideal)
        ndcgs.append(_dcg(gains) / idcg if idcg > 0 else 0.0)

    return {
        "Precision@%d" % k: float(np.mean(precisions)) if precisions else 0.0,
        "Recall@%d" % k: float(np.mean(recalls)) if recalls else 0.0,
        "NDCG@%d" % k: float(np.mean(ndcgs)) if ndcgs else 0.0,
    }


def coverage(recommended_lists, catalog_size: int) -> float:
    """Fraction of the catalog that appears across all users' recommendations."""
    rec = set()
    for lst in recommended_lists:
        rec.update(lst)
    return len(rec) / catalog_size if catalog_size else 0.0


def compare_cf_models(train: pd.DataFrame, test: pd.DataFrame,
                      models: dict, k: int = 10, threshold: float = 4.0) -> pd.DataFrame:
    """RMSE + Precision/Recall/NDCG@K for each model in {name: model}.

    models may be CFModel, PopularityModel, or any object with predict() and
    predict_for_user().
    """
    rows = []
    for name, model in models.items():
        # Close over each model explicitly so the metric helper sees a uniform
        # score_user_items(user_id, book_ids) callable.
        def scorer(m):
            def f(user_id, book_ids):
                return m.predict_for_user(user_id, book_ids)
            return f

        metrics = ranking_metrics(scorer(model), train, test, k=k, threshold=threshold)
        metrics["RMSE"] = round(rmse(model, test), 4)
        metrics["model"] = name
        rows.append(metrics)
    out = pd.DataFrame(rows).set_index("model")
    cols = ["RMSE"] + [c for c in out.columns if c != "RMSE"]
    return out[cols].round(4)
