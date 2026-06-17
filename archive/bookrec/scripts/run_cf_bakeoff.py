#!/usr/bin/env python3
"""
Project 2 — collaborative-filtering model bake-off.

Reproduces the teammate's Project 2 shell using the shared `src/` package, and
demonstrates the model the app ships. Trains the baseline plus the class
neighborhood models AND the corrected-audit winners (pearson_baseline) on a
90/10 hold-out split, then prints RMSE + Precision@10 + Recall@10 + NDCG@10.

Precision@10 uses the CORRECTED denominator (min(k, held-out count); see
evaluate.ranking_metrics and notebooks/precision_at_k_corrected.ipynb). Under
that metric IBCF pearson_baseline (k=9) wins on F1 and RMSE — the config wired
into app.py.

Run from the bookrec folder:
    python scripts/run_cf_bakeoff.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import cf_model, data_loader, evaluate, recommend


RANDOM_STATE = 6604
K_NEIGHBORS = 10
MIN_RATINGS = 20
TOP_N = 5
DEMO_USER = 314


def main() -> None:
    ratings, books = data_loader.load("data")
    print(f"Books: {books.shape}  |  Ratings: {ratings.shape}")

    train, test = evaluate.train_test_split_ratings(
        ratings, test_size=0.1, seed=RANDOM_STATE
    )
    print(f"Train: {len(train)} ratings  |  Test: {len(test)} ratings")

    models = {
        # BaselineOnly is the Surprise global/user/item mean benchmark; the
        # pearson/cosine pair are the original class-notebook neighborhood models;
        # the pearson_baseline pair are the corrected-audit tuned winners. IBCF
        # pearson_baseline is what app.py ships.
        "Baseline": cf_model.CFModel("baseline").fit(train),
        "UBCF pearson": cf_model.CFModel("ubcf", k=K_NEIGHBORS).fit(train),
        "IBCF cosine": cf_model.CFModel("ibcf", k=K_NEIGHBORS).fit(train),
        "UBCF pearson_baseline k=25":
            cf_model.CFModel("ubcf", k=25, sim_name="pearson_baseline").fit(train),
        "IBCF pearson_baseline k=9 (app)":
            cf_model.CFModel("ibcf", k=9, sim_name="pearson_baseline").fit(train),
    }

    results = evaluate.compare_cf_models(train, test, models, k=10, threshold=4.0)
    results = results.sort_values("Precision@10", ascending=False)
    print("\nModel comparison (hold-out test set, corrected Precision@10):")
    print(results.to_string())

    if DEMO_USER in set(ratings["user_id"]):
        ubcf = models["UBCF pearson"]
        ibcf = models["IBCF pearson_baseline k=9 (app)"]
        label_of = data_loader.title_lookup(books)

        def top_n(model, user_id: int, n: int = TOP_N):
            class Adapter:
                # recommend_top_n expects a hybrid-like .score method; this
                # adapter lets plain CF models plug into the same path.
                def score(self, uid, user_ratings, candidate_ids):
                    return model.predict_for_user(uid, candidate_ids)

            recs = recommend.recommend_top_n(
                user_id,
                Adapter(),
                ratings,
                books,
                top_n=n,
                min_ratings=MIN_RATINGS,
                explain=False,
            )
            return [(label_of.get(r.book_id, r.title), r.score) for r in recs.itertuples()]

        print(f"\nTop-{TOP_N} UBCF recommendations for user {DEMO_USER}:")
        for title, score in top_n(ubcf, DEMO_USER):
            print(f"  {score:.3f}  {title}")

        print(f"\nTop-{TOP_N} IBCF recommendations for user {DEMO_USER}:")
        for title, score in top_n(ibcf, DEMO_USER):
            print(f"  {score:.3f}  {title}")


if __name__ == "__main__":
    main()
