# 📚 bookrec — Goodreads Book Recommender (project scaffold)

A runnable starting point for your recommendation project. It extends the
Week-3 class notebook (collaborative filtering with `surprise`) into a
**hybrid book recommender** (collaborative + content) with a **Streamlit app**.

It runs **on day one with synthetic data** — no downloads needed — so you can
see the whole pipeline work, then swap in real goodbooks-10k data.

## Quickstart

```bash
cd bookrec
pip install -r requirements.txt          # numpy<2.0 matters for scikit-surprise
streamlit run app.py                     # launches the app in your browser
```

Or work through the notebook first: `notebooks/01_build_book_recommender.ipynb`.

## What's here

```
bookrec/
  app.py                       # Streamlit app (the deliverable)
  requirements.txt
  PROJECT_FRAMEWORK.md         # the plan: milestones, architecture, decisions
  README.md                    # this file
  data/
    README.md                  # where to put goodbooks-10k (gotchas included)
  notebooks/
    01_build_book_recommender.ipynb   # bridges the class notebook -> the project
  src/
    sample_data.py             # synthetic data so it runs before any download
    data_loader.py             # load/clean ratings + book metadata
    cf_model.py                # collaborative filtering (your class model, modularized)
    content_model.py           # content-based similarity (TF-IDF; optional embeddings)
    hybrid.py                  # blend CF + content; cold start
    evaluate.py                # RMSE + Precision/Recall@K + NDCG@K + coverage
    recommend.py               # top-N orchestration + explanations
```

## The one mental model

Everything flows the same way as class, just with more stages:

```
ratings + book metadata
        │
        ├─► CF model (surprise)         → predicts a rating  (what you know)
        ├─► content model (TF-IDF/embeds)→ similarity to your taste (new)
        │
        └─► hybrid blend ──► top-N ──► Streamlit app
                              │
                         evaluate: RMSE, Precision/Recall@K, NDCG@K, coverage
```

See `PROJECT_FRAMEWORK.md` for the build order and `../goodreads_recommender_guide.md`
for the deep theory behind each stage.
