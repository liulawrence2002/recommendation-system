"""
Content-based model — the "go beyond class" half.

Builds a vector per book from its metadata (title + authors + shelf tags),
then recommends by similarity to the books a user already liked. This is
what gives the system: (a) cold-start coverage for brand-new books, and
(b) "more like this" explanations.

Default vectorizer is TF-IDF (fast, no heavy deps, mirrors the guide's
Section 6.1). To upgrade to semantic embeddings (guide Section 6.2), set
backend="embeddings" and install sentence-transformers — the rest of the
package is unchanged because both backends expose the same .item_vectors.

Vectors are kept in memory (numpy/sklearn sparse). At ~10k books this is
sufficient; do not add ChromaDB unless the catalog grows to millions of items.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


def _content_document(books: pd.DataFrame) -> pd.Series:
    """One text blob per book. Repeat tags so shelves dominate the signal."""
    title = books["title"].fillna("").astype(str)
    authors = books["authors"].fillna("").astype(str)
    tags = books["tags"].fillna("").astype(str)
    return title + " " + authors + " " + tags + " " + tags  # tags weighted x2


class ContentModel:
    def __init__(self, backend: str = "tfidf", embed_model: str = "all-MiniLM-L6-v2"):
        self.backend = backend
        self.embed_model = embed_model
        self.books = None
        self.item_vectors = None          # [n_books, d], L2-normalized
        self._row_of = {}                 # book_id -> row index

    def fit(self, books: pd.DataFrame):
        self.books = books.reset_index(drop=True)
        self._row_of = {bid: i for i, bid in enumerate(self.books["book_id"])}
        docs = _content_document(self.books)

        if self.backend == "tfidf":
            self.vectorizer = TfidfVectorizer(
                ngram_range=(1, 2), min_df=1, max_df=0.9,
                stop_words="english", sublinear_tf=True)
            X = self.vectorizer.fit_transform(docs)        # sparse, already L2-normalized
            self.item_vectors = normalize(X)               # be explicit
        elif self.backend == "embeddings":
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(self.embed_model)
            emb = model.encode(list(docs), normalize_embeddings=True,
                               show_progress_bar=False)
            self.item_vectors = normalize(np.asarray(emb))
        else:
            raise ValueError(f"unknown backend: {self.backend!r}")
        return self

    # --- item-item: "because you liked X" ----------------------------------
    def similar_books(self, book_id, k: int = 10):
        if book_id not in self._row_of:
            return []
        row = self._row_of[book_id]
        sims = (self.item_vectors @ self.item_vectors[row].T)
        sims = np.asarray(sims.todense()).ravel() if hasattr(sims, "todense") else np.asarray(sims).ravel()
        order = np.argsort(-sims)
        out = [(self.books.iloc[i]["book_id"], float(sims[i])) for i in order if i != row]
        return out[:k]

    # --- grounding: candidate -> the user's own favorite it most resembles --
    def nearest_examples(self, candidate_ids, liked_ids):
        """For each candidate, the single most content-similar book among the
        user's ``liked_ids`` (books they rated highly).

        Returns ``{candidate_id: (liked_book_id, similarity)}``, skipping any
        candidate with no positive match. Used to *ground* the LLM re-ranker's
        explanations in the reader's real history ("in the spirit of X, which you
        rated highly") instead of letting the model guess from the title alone.
        """
        out = {}
        liked = [(b, self._row_of[b]) for b in liked_ids if b in self._row_of]
        if not liked or self.item_vectors is None:
            return out
        lids, lidx = zip(*liked)
        L = self.item_vectors[list(lidx)]
        for b in candidate_ids:
            r = self._row_of.get(b)
            if r is None:
                continue
            v = self.item_vectors[r]
            sims = L @ (v.T if hasattr(v, "T") else v)
            sims = (np.asarray(sims.todense()).ravel()
                    if hasattr(sims, "todense") else np.asarray(sims).ravel())
            order = np.argsort(-sims)
            for j in order:                      # take the best match that isn't itself
                if lids[j] != b and sims[j] > 0:
                    out[b] = (lids[j], float(sims[j]))
                    break
        return out

    # --- user profile = liked-book vectors, rating-weighted -----------------
    def user_profile(self, user_ratings: pd.DataFrame, like_threshold: float = 4.0):
        """Mean of the vectors of books the user rated >= threshold (rating-weighted)."""
        liked = user_ratings[user_ratings["rating"] >= like_threshold]
        rows = [self._row_of[b] for b in liked["book_id"] if b in self._row_of]
        if not rows:
            return None
        w = liked["rating"].values.reshape(-1, 1)
        V = self.item_vectors[rows]
        prof = (V.multiply(w) if hasattr(V, "multiply") else V * w)
        prof = np.asarray(prof.sum(axis=0)).ravel() if hasattr(prof, "sum") else prof.sum(axis=0)
        n = np.linalg.norm(prof)
        return prof / n if n > 0 else None

    def score_user(self, user_ratings: pd.DataFrame, book_ids=None) -> pd.Series:
        """Content score in [0,1] (cosine to the user's taste profile) per book."""
        if book_ids is None:
            book_ids = list(self.books["book_id"])
        prof = self.user_profile(user_ratings)
        if prof is None:                               # cold user: no liked books yet
            return pd.Series(0.0, index=book_ids, name="content_score")
        rows = [self._row_of.get(b) for b in book_ids]
        valid = [(b, r) for b, r in zip(book_ids, rows) if r is not None]
        if not valid:
            return pd.Series(0.0, index=book_ids, name="content_score")
        bids, ridx = zip(*valid)
        V = self.item_vectors[list(ridx)]
        sims = np.asarray(V @ prof).ravel()
        sims = (sims + 1) / 2.0                         # cosine [-1,1] -> [0,1]
        return pd.Series(sims, index=list(bids), name="content_score")
