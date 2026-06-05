# Data folder

**The assignment data is already here:** `Books.csv` and `Ratings.csv`. Load it with:

```python
from src.data_loader import load
ratings, books = load("data")     # reads Books.csv + Ratings.csv
```

Schema (data dictionary):
- `Books.csv`: book_id, isbn, authors, original_publication_year, title,
  language_code, average_rating, ratings_count, text_reviews_count,
  ratings_1..ratings_5, image_url, small_image_url
- `Ratings.csv`: book_id, user_id, rating (1–5)

Notes: there is **no shelf-tags file** in this dataset (content signal is
title + authors only), and some author names arrive mojibaked
(`Mary GrandPrÃ©`) — `load()` repairs the encoding automatically.

**No vector database:** content similarity is computed in memory (`src/content_model.py`).
This catalog (~10k books) does not need ChromaDB/FAISS for the assignment pipeline.

---

The framework also runs **without any data** using a synthetic generator
(`src/sample_data.py`, via `data_loader.load_sample()`) for offline development.
The multi-file `load_goodbooks()` loader below is for the original
goodbooks-10k release (with a tags file), if you ever use it instead.

## Get goodbooks-10k

Download from https://github.com/zygmuntz/goodbooks-10k (or the Kaggle mirror)
and place these files in this folder:

```
bookrec/data/
  ratings.csv      # user_id, book_id, rating (1..5)   — ~6M rows
  books.csv        # book_id, goodreads_book_id, work_id, authors, title, ...
  book_tags.csv    # goodreads_book_id, tag_id, count   (shelf tags)
  tags.csv         # tag_id, tag_name
  to_read.csv      # user_id, book_id  (optional implicit signal)
```

Then in the notebook / app, switch the loader:

```python
from src.data_loader import load_goodbooks
ratings, books = load_goodbooks("data")   # instead of load_sample()
```

## Schema gotchas (read before you debug)

- `book_tags.csv` is keyed on **`goodreads_book_id`**, not `book_id`.
  `load_goodbooks` maps it for you via `books.csv`.
- `book_id` (1..10000) is the join key for ratings; `work_id` deduplicates
  editions of the same title — dedupe on it if you want one row per work.
- goodbooks-10k has **no descriptions**. Content features come from
  title + authors + shelf tags. For richer text, use the
  `goodbooks-10k-extended` repo or the UCSD Goodreads dataset (has reviews).

## Bigger / richer alternatives

- **UCSD Goodreads "Book Graph"** (Wan & McAuley) — ~2.36M books, ~229M
  interactions, full review text, series/authors graph. Academic use only.
  https://mengtingwan.github.io/data/goodreads.html
