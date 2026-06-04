# Project 2 — Goodreads Book Recommender: Plan

Refined against the actual assignment brief and rubric (100 pts). This is the
working plan for the **code first**; the slide/video deliverables come later.

---

## 1. Problem statement

Explore the Goodreads dataset, build and compare **collaborative-filtering**
recommenders against a **popularity/mean baseline**, then add an **LLM
re-ranking layer** that personalizes the Top-N to a user's stated preference.
Wrap it in a **Streamlit app** and present findings in a slide deck + video demo.

Domain facts that shape the approach: explicit 1–5 star ratings (so RMSE is
meaningful), stable preferences, a text-light catalog (no shelf tags here), and
a strong popularity signal — which the brief explicitly warns may be hard for CF
to beat on this sparse data.

---

## 2. The data (reproduce from these two files only)

`data/Books.csv` (~9,964 books) and `data/Ratings.csv` (~164,728 ratings).

- **Books**: `book_id, isbn, authors, original_publication_year, title,
  language_code, average_rating, ratings_count, text_reviews_count,
  ratings_1..ratings_5, image_url, small_image_url`.
- **Ratings**: `book_id, user_id, rating` (1–5).
- **No tags/shelf file** → content signal is limited to title + authors. This is
  why the **LLM layer (not a content model) is the personalization step**.
- **Encoding quirk**: some author names are mojibake (`Mary GrandPrÃ©`);
  `data_loader.load()` repairs this. Flag it as an EDA/data-cleaning finding.

Load it with the assignment-spec loader:

```python
from src.data_loader import load
ratings, books = load("data")     # reads Books.csv + Ratings.csv
```

Reproducibility requirement: the whole pipeline must run from just these two
CSVs. No API key in the submission (the LLM key is read from the environment).

---

## 3. Deliverables & rubric map

| Rubric criterion | Pts | Where it's earned | Status in scaffold |
|---|---|---|---|
| EDA & insights | 10 | notebook §EDA | **TODO** (add EDA cells) |
| CF modeling & evaluation | 30 | `cf_model.py`, `evaluate.py`, notebook | ✅ models + metrics built |
| LLM personalization layer | 20 | `llm_rerank.py`, app | ✅ module + fallback built |
| Streamlit app & video | 15 | `app.py` | ✅ app built; video TODO |
| Slides & communication | 15 | (deliverable) | TODO |
| Code quality & reproducibility | 5 | whole `src/` | ✅ runs from the 2 CSVs |
| Creative design / extra | 5 | content hybrid, UX, explanations | ✅ hybrid available as extra |

Slide deck rules: **PDF, ≤ 8 slides** (title slide with team #/names + up to 2
appendix slides), **font ≥ 16**. Video: **~2–3 min**, shareable link (Zoom cloud),
walk through CF step, LLM step, and one design choice.

---

## 4. Required analyses (the graded substance)

**EDA (10 pts).** Users (ratings-per-user distribution, sparsity), ratings
(distribution, skew toward 4–5 stars = MNAR hint), books (popularity long tail,
publication-year trends, `average_rating` vs `ratings_count`, language mix,
the encoding anomaly). Call out what each implies for modeling (sparsity →
CF may struggle; popularity skew → strong baseline).

**CF modeling & evaluation (30 pts).**
- Models: **popularity/mean baseline**, **user-based CF (Pearson)**,
  **item-based CF (cosine)** — all in `surprise` (+ SVD available as a bonus).
- Hold-out evaluation: **RMSE** (rating accuracy) and **Precision@N / Recall@N**
  (Top-N). Use a reproducible split (`random_state`).
- Compare and **interpret**: which wins and *why*. The brief explicitly invites
  the finding that **CF may not beat a strong popularity baseline on sparse data**
  — if so, say so and explain (sparsity, cold items, popularity dominance), and
  state what that implies for model selection. That honest interpretation scores.

**LLM personalization layer (20 pts).**
- Take Top-N from the **best CF model**, pass book **metadata** (title, authors,
  year, average_rating) as context, and have the LLM **re-rank** + explain to a
  stated preference (mood/genre). It must **not invent books** — `llm_rerank.py`
  enforces this by accepting only candidate `book_id`s.
- Prompt strategy: structured JSON output, explicit "choose only from these ids,"
  one-sentence preference-grounded explanation each. Experiment with prompt
  styles and note what worked. Cite **provider + model** (Gemini recommended,
  free; e.g. `gemini-2.0-flash`).

**Business discussion.** Applications of CF (scalable, behavior-driven, cold-start
weak) vs the LLM layer (flexible, explainable, controllable, but cost/latency and
hallucination risk). Setup challenges (data sparsity, popularity bias, API cost,
keys/privacy). Your recommended approach for *this* dataset.

---

## 5. How the code maps (what exists vs what's left)

```
src/
  data_loader.py   ✅ load("data") reads Books.csv + Ratings.csv (+ encoding fix)
  cf_model.py      ✅ PopularityModel + CFModel(ubcf/ibcf/svd/baseline)
  evaluate.py      ✅ RMSE + Precision/Recall/NDCG@K + coverage + split
  recommend.py     ✅ Top-N orchestration (popularity filter, exclude seen)
  llm_rerank.py    ✅ Gemini re-rank + explanations + heuristic fallback
  content_model.py ⭐ optional EXTRA (content hybrid; not the required layer)
  hybrid.py        ⭐ optional EXTRA
app.py             ✅ user → CF Top-N → LLM re-rank
notebooks/01_...   ⚠️ update to use load() and the CF→LLM flow; ADD EDA cells
```

Left to do in code: (1) EDA cells in the notebook; (2) a clean model-comparison
table on the real data; (3) point the notebook at `load()`; (4) try a real
Gemini key end-to-end (optional locally).

---

## 6. Milestones

- **M0 (done)** scaffold runs; real data loaded.
- **M1 — EDA.** Add notebook EDA cells + 3–4 charts; capture insights.
- **M2 — CF comparison.** Train baseline/UBCF/IBCF; produce the RMSE + P@N/R@N
  table on the real data; write the interpretation.
- **M3 — LLM layer.** Get a Gemini key (env var), test `llm_rerank.rerank` live;
  iterate on the prompt; capture 2–3 example outputs for the slides.
- **M4 — App.** Polish `app.py`; test multiple users/preferences.
- **M5 — Deliverables.** Slides (≤8), record the 2–3 min demo, final code pass
  (annotations, no key, reproducible from the 2 CSVs).

---

## 7. Decisions / risks to note in the writeup

- **CF vs baseline.** Likely close on sparse data — frame the comparison honestly.
- **Split.** Random hold-out (no timestamps in this data); state the limitation.
- **Rating vs ranking.** Train/score on ratings (RMSE) but also report Top-N
  ranking metrics; note they can disagree.
- **LLM cost/keys.** Read key from env; never commit it; fallback keeps the app
  runnable and the code reproducible without a key.
- **Popularity bias.** The `min_ratings` filter and popularity baseline both lean
  on popular books — watch coverage and mention the trade-off.

Theory backup for any section: `../goodreads_recommender_guide.md`.
