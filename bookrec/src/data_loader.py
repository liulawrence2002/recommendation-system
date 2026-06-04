"""
Load and clean data into the package's column contract:

    ratings : DataFrame[user_id, book_id, rating]
    books   : DataFrame[book_id, title, authors, original_publication_year,
                        average_rating, ratings_count, tags]

Primary entry point for THIS project:
    load(data_dir) -> reads the assignment's Books.csv + Ratings.csv.

Also available:
    load_sample()       -> synthetic data (no downloads needed).
    load_goodbooks(dir) -> the original multi-file goodbooks-10k release.
"""
from __future__ import annotations

import os
import pandas as pd

REQUIRED_BOOK_COLS = ["book_id", "title", "authors", "original_publication_year",
                      "average_rating", "ratings_count", "tags"]


def load_sample():
    """Synthetic data for development. See sample_data.make_sample."""
    from .sample_data import make_sample
    return make_sample()


def _fix_mojibake(s):
    """Repair UTF-8 text mis-decoded as latin-1 (e.g. 'GrandPrÃ©' -> 'GrandPré')."""
    if not isinstance(s, str):
        return s
    try:
        return s.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s


def load(data_dir: str = "data", fix_encoding: bool = True):
    """Primary loader: read the assignment Books.csv + Ratings.csv.

    Reproducible from just those two files. There is no shelf-tags file in this
    dataset, so an empty `tags` column is added and the content model falls back
    to title + authors. Returns (ratings, books) in the package contract.
    """
    def _path(name):
        p = os.path.join(data_dir, name)
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Could not find {name} in {data_dir!r}. "
                f"Place Books.csv and Ratings.csv there (see bookrec/data/README.md)."
            )
        return p

    books = pd.read_csv(_path("Books.csv"))
    ratings = pd.read_csv(_path("Ratings.csv"))

    if fix_encoding:
        for col in ("authors", "title"):
            if col in books.columns:
                books[col] = books[col].map(_fix_mojibake)

    books["tags"] = ""                       # no shelf tags in this dataset
    books = _coerce_books(books)
    ratings = ratings[["user_id", "book_id", "rating"]].copy()
    ratings["rating"] = ratings["rating"].astype(float)
    return ratings, books


def load_goodbooks(data_dir: str, top_tags_per_book: int = 10):
    """Load the original multi-file goodbooks-10k release (with a tags file)."""
    def _path(name):
        p = os.path.join(data_dir, name)
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Could not find {name} in {data_dir!r}. See bookrec/data/README.md."
            )
        return p

    ratings = pd.read_csv(_path("ratings.csv"))
    books = pd.read_csv(_path("books.csv"))
    try:
        book_tags = pd.read_csv(_path("book_tags.csv"))
        tags = pd.read_csv(_path("tags.csv"))
        bt = book_tags.merge(tags, on="tag_id", how="left")
        bt = (bt.sort_values(["goodreads_book_id", "count"], ascending=[True, False])
                .groupby("goodreads_book_id").head(top_tags_per_book))
        tags_str = (bt.groupby("goodreads_book_id")["tag_name"]
                      .apply(lambda s: " ".join(str(x).replace("-", "") for x in s))
                      .rename("tags"))
        books = books.merge(tags_str, left_on="goodreads_book_id",
                            right_index=True, how="left")
    except FileNotFoundError:
        pass
    if "tags" not in books.columns:
        books["tags"] = ""
    books["tags"] = books["tags"].fillna("")
    books = _coerce_books(books)
    ratings = ratings[["user_id", "book_id", "rating"]].copy()
    ratings["rating"] = ratings["rating"].astype(float)
    return ratings, books


def _coerce_books(books: pd.DataFrame) -> pd.DataFrame:
    """Ensure all contract columns exist with safe defaults."""
    defaults = {
        "title": "Untitled", "authors": "Unknown",
        "original_publication_year": 0, "average_rating": 0.0,
        "ratings_count": 0, "tags": "",
    }
    for col, val in defaults.items():
        if col not in books.columns:
            books[col] = val
        books[col] = books[col].fillna(val)
    return books


def title_lookup(books: pd.DataFrame) -> dict:
    """book_id -> 'Title - Author' for display."""
    return {row.book_id: f"{row.title} - {row.authors}"
            for row in books.itertuples(index=False)}
