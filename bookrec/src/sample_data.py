"""
Synthetic Goodreads-style data so the framework runs BEFORE you download
the real goodbooks-10k files.

It fabricates books that belong to latent "genres" and users who have a
taste for some genres, so that BOTH collaborative and content signals are
real and learnable. Swap this out for `data_loader.load_goodbooks(...)`
once you've dropped the real CSVs into bookrec/data/.

Returned frames match the column contract the rest of the package expects:

    ratings : DataFrame[user_id, book_id, rating]          # rating in 1..5
    books   : DataFrame[book_id, title, authors,
                        original_publication_year,
                        average_rating, ratings_count, tags]  # tags = space-joined string
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GENRES = ["fantasy", "scifi", "romance", "mystery", "history",
          "horror", "literary", "ya", "thriller", "nonfiction"]


def make_sample(n_users: int = 400, n_books: int = 600,
                n_ratings: int = 12000, seed: int = 6604):
    """Generate (ratings, books) with latent-genre structure.

    Each book gets 1-2 genres and a fake author. Each user has a preference
    weight per genre; their rating of a book is driven by genre match plus
    noise, then clipped to the 1..5 star scale. This guarantees that a good
    model can actually do better than random.
    """
    rng = np.random.default_rng(seed)

    # --- books: each book belongs to one or two genres ----------------------
    # The generated titles/tags intentionally expose the latent genre signal so
    # the content model has something meaningful to learn during demos.
    book_genres, authors, titles, tags = [], [], [], []
    for b in range(n_books):
        k = rng.integers(1, 3)                       # 1 or 2 genres
        gs = list(rng.choice(GENRES, size=k, replace=False))
        book_genres.append(gs)
        authors.append(f"Author {rng.integers(0, n_books // 4)}")
        titles.append(f"Book {b}: {' & '.join(gs).title()} Story")
        # tags = genres repeated a bit to mimic Goodreads shelf-tag bag-of-words
        tag_tokens = []
        for g in gs:
            tag_tokens += [g] * int(rng.integers(2, 6))
        tags.append(" ".join(tag_tokens))

    books = pd.DataFrame({
        "book_id": np.arange(n_books),
        "title": titles,
        "authors": authors,
        "original_publication_year": rng.integers(1950, 2024, size=n_books),
        "average_rating": np.round(rng.uniform(3.2, 4.6, size=n_books), 2),
        "ratings_count": rng.integers(20, 50000, size=n_books),
        "tags": tags,
    })

    # genre membership matrix [n_books, n_genres] for rating simulation
    g_index = {g: i for i, g in enumerate(GENRES)}
    book_mat = np.zeros((n_books, len(GENRES)))
    for b, gs in enumerate(book_genres):
        for g in gs:
            book_mat[b, g_index[g]] = 1.0

    # --- users: each has a latent taste vector over genres ------------------
    # Positive/negative weights create user-specific taste profiles rather than
    # one global popularity pattern.
    user_taste = rng.normal(size=(n_users, len(GENRES)))

    # --- ratings: sample (user, book) pairs, rate by taste . genre + noise --
    u = rng.integers(0, n_users, size=n_ratings)
    b = rng.integers(0, n_books, size=n_ratings)
    affinity = (user_taste[u] * book_mat[b]).sum(axis=1)
    # squash affinity into a 1..5 star scale with noise
    stars = 3.0 + 0.9 * affinity + rng.normal(scale=0.7, size=n_ratings)
    stars = np.clip(np.round(stars), 1, 5)

    ratings = (pd.DataFrame({"user_id": u, "book_id": b, "rating": stars})
               # Collapse duplicate sampled pairs so the synthetic data matches
               # the one-rating-per-user-book contract used by real Goodreads data.
               .drop_duplicates(subset=["user_id", "book_id"])   # one rating per pair
               .reset_index(drop=True))
    ratings["rating"] = ratings["rating"].astype(float)

    return ratings, books


if __name__ == "__main__":
    r, bk = make_sample()
    print(f"sample ratings: {r.shape}  | sample books: {bk.shape}")
    print(r.head())
    print(bk.head())
