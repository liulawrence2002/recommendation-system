"""
Project 3 — build the Braintrust evaluation dataset from the Project 2 recommender.

What this does
--------------
1. Loads the same Goodreads data Project 2 uses (Books.csv + Ratings.csv).
2. Trains the SAME best collaborative-filtering model the app ships:
       UBCF (user-based KNN), similarity = "pearson_baseline", k = 25.
3. Picks 10 users (reproducible, seeded).
4. For each user, generates the Top-10 CF candidates with exactly the fields the
   re-ranking prompt sees: book_id, title, authors, year, average_rating, cf_score.
5. Pairs every user with the SAME two contrasting requests (mood vs. character),
   producing 20 cases (10 users x 2 requests). Each case is one row.
6. Writes three artifacts into ../data/:
       eval_dataset.csv    - human-readable, one row per case (candidates as JSON)
       eval_dataset.jsonl  - Braintrust-friendly (input + metadata per line)
       candidates_preview.txt - the 10 candidates per user, as plain text to eyeball

Run from the `project 3/scripts` folder (or anywhere — paths are resolved
relative to this file):

    pip install scikit-surprise pandas
    python build_eval_dataset.py
"""
from __future__ import annotations

import json
import os
import sys

import pandas as pd

# ---------------------------------------------------------------------------
# Paths: reuse the Project 2 pipeline in ../../bookrec/src
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
BOOKREC = os.path.join(REPO, "bookrec")
DATA_DIR = os.path.join(BOOKREC, "data")
OUT_DIR = os.path.abspath(os.path.join(HERE, "..", "data"))
sys.path.insert(0, BOOKREC)

from src.data_loader import load                       # noqa: E402
from src.cf_model import CFModel                        # noqa: E402
from src.recommend import popular_books                 # noqa: E402

# ---------------------------------------------------------------------------
# Configuration — the knobs that define the dataset
# ---------------------------------------------------------------------------
N_USERS = 10
TOP_N = 10
MIN_RATINGS = 20          # popularity floor, same as recommend_top_n default
SEED = 6604               # reproducible user sample (matches Project 2 random_state)

# Best CF model from Project 2's corrected Top-N audit.
CF_KIND = "ubcf"
CF_SIM = "pearson_baseline"
CF_K = 25

# The two contrasting requests — the SAME two used for every user.
REQUESTS = [
    {
        "request_type": "mood",        # mood / tone
        "request": "Something light, funny, and fast-paced to unwind with after a long week.",
    },
    {
        "request_type": "character",   # content / character
        "request": "Something thoughtful with strong, complex character development that stays with me.",
    },
]


def pick_users(ratings: pd.DataFrame, n: int, seed: int) -> list:
    """Reproducible sample of n users that have enough history for solid CF.

    Every user in this dataset has 100-200 ratings, so any user is a fine pick;
    we sample with a fixed seed so the dataset is regenerable bit-for-bit."""
    counts = ratings["user_id"].value_counts()
    eligible = counts[counts >= 50].index.to_series()
    return sorted(eligible.sample(n=n, random_state=seed).tolist())


def candidates_for_user(user_id, cf, ratings, books, top_n, min_ratings) -> list:
    """Replicate recommend_top_n's candidate logic, returning the reranker fields.

    - exclude books the user has already rated
    - keep only books above the popularity floor (relax if that empties the set)
    - score with the CF model, take the Top-N
    - attach title/authors/year/average_rating/cf_score (what the prompt uses)
    """
    seen = set(ratings.loc[ratings["user_id"] == user_id, "book_id"])
    pop = popular_books(ratings, min_ratings)
    unseen = [b for b in books["book_id"] if b not in seen]
    cands = [b for b in unseen if b in pop] or unseen

    scores = cf.predict_for_user(user_id, cands)
    top = scores.sort_values(ascending=False).head(top_n)

    meta = books.set_index("book_id")
    out = []
    for book_id, sc in top.items():
        row = meta.loc[book_id]
        year = row.get("original_publication_year", 0)
        year = int(float(year)) if year and float(year) == float(year) else "?"
        avg = row.get("average_rating", 0.0)
        avg = round(float(avg), 2) if avg and float(avg) == float(avg) else "?"
        out.append({
            "book_id": int(book_id),
            "title": str(row.get("title", "")),
            "authors": str(row.get("authors", "")),
            "year": year,
            "average_rating": avg,
            "cf_score": round(float(sc), 3),
        })
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Loading data from {DATA_DIR} ...")
    ratings, books = load(DATA_DIR)
    print(f"  {len(ratings):,} ratings  {books['book_id'].nunique():,} books  "
          f"{ratings['user_id'].nunique():,} users")

    print(f"Training {CF_KIND.upper()} (sim={CF_SIM}, k={CF_K}) on all ratings ...")
    cf = CFModel(kind=CF_KIND, k=CF_K, sim_name=CF_SIM).fit(ratings)

    users = pick_users(ratings, N_USERS, SEED)
    print(f"Selected {len(users)} users: {users}")

    rows, jsonl, preview = [], [], []
    case_id = 0
    for uid in users:
        cands = candidates_for_user(uid, cf, ratings, books, TOP_N, MIN_RATINGS)

        preview.append(f"\n=== user {uid} — top {len(cands)} CF candidates ===")
        for i, c in enumerate(cands, 1):
            preview.append(f"  {i:>2}. [{c['book_id']}] {c['title']} — {c['authors']} "
                           f"({c['year']}), avg={c['average_rating']}, cf={c['cf_score']}")

        for req in REQUESTS:
            case_id += 1
            rows.append({
                "case_id": case_id,
                "user_id": uid,
                "request_type": req["request_type"],
                "request": req["request"],
                "candidates_json": json.dumps(cands, ensure_ascii=False),
            })
            jsonl.append({
                "input": {"request": req["request"], "candidates": cands},
                "metadata": {
                    "case_id": case_id,
                    "user_id": int(uid),
                    "request_type": req["request_type"],
                },
            })

    # CSV (one row per case)
    csv_path = os.path.join(OUT_DIR, "eval_dataset.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    # JSONL (Braintrust)
    jsonl_path = os.path.join(OUT_DIR, "eval_dataset.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in jsonl:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Preview text
    prev_path = os.path.join(OUT_DIR, "candidates_preview.txt")
    with open(prev_path, "w", encoding="utf-8") as f:
        f.write("\n".join(preview))

    print(f"\nWrote {len(rows)} cases:")
    print(f"  {csv_path}")
    print(f"  {jsonl_path}")
    print(f"  {prev_path}")


if __name__ == "__main__":
    main()
