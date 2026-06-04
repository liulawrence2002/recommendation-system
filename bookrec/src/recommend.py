"""
Top-N orchestration used by the notebook and the Streamlit app.

Mirrors the class notebook's `top_n_for_user`, but:
  * works for CF, content, or hybrid models (anything with the right interface),
  * applies a popularity filter (>= MIN_RATINGS) like class,
  * excludes books the user has already rated,
  * attaches a short "why" explanation from the content model.
"""
from __future__ import annotations

import pandas as pd


def popular_books(ratings: pd.DataFrame, min_ratings: int = 20) -> set:
    counts = ratings["book_id"].value_counts()
    return set(counts[counts >= min_ratings].index)


def recommend_top_n(user_id, hybrid, ratings: pd.DataFrame, books: pd.DataFrame,
                    top_n: int = 10, min_ratings: int = 20,
                    explain: bool = True) -> pd.DataFrame:
    """Return a DataFrame of the user's top-N recommended books."""
    seen = set(ratings.loc[ratings["user_id"] == user_id, "book_id"])
    pop = popular_books(ratings, min_ratings)
    candidates = [b for b in books["book_id"] if b not in seen and b in pop]
    if not candidates:                          # tiny catalog / cold start fallback
        candidates = [b for b in books["book_id"] if b not in seen]

    user_ratings = ratings[ratings["user_id"] == user_id]
    scores = hybrid.score(user_id, user_ratings, candidates)
    top = scores.sort_values(ascending=False).head(top_n)

    title_of = dict(zip(books["book_id"], books["title"]))
    author_of = dict(zip(books["book_id"], books["authors"]))
    rows = []
    for book_id, sc in top.items():
        row = {"book_id": book_id, "title": title_of.get(book_id, "?"),
               "authors": author_of.get(book_id, "?"), "score": round(float(sc), 4)}
        if explain and getattr(hybrid, "content", None) is not None:
            row["why"] = _explain(hybrid, user_ratings, book_id, title_of)
        rows.append(row)
    return pd.DataFrame(rows)


def _explain(hybrid, user_ratings, book_id, title_of) -> str:
    """Cheap content-based 'because you liked ...' string."""
    liked = user_ratings[user_ratings["rating"] >= 4.0]["book_id"].tolist()
    if not liked or hybrid.content is None:
        return "popular in your taste profile"
    # find which liked book is most similar to this recommendation
    sims = dict(hybrid.content.similar_books(book_id, k=50))
    best = max(liked, key=lambda b: sims.get(b, 0.0)) if liked else None
    if best is not None and sims.get(best, 0.0) > 0:
        return f"because you liked “{title_of.get(best, best)}”"
    return "matches your genres"
