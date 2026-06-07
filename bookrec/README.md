# bookrec — Goodreads Book Recommender (Project 2)

Collaborative-filtering recommender for the Goodreads assignment dataset, with a
**Streamlit app** and optional **Gemini LLM re-ranking** layer.

## Quickstart

```bash
cd bookrec
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt # numpy<2.0 required for scikit-surprise
streamlit run app.py
```

**CLI bake-off** (teammate's Project 2 shell):

```bash
python scripts/run_cf_bakeoff.py
```

Or work through the notebook: `notebooks/01_build_book_recommender.ipynb`.

## What's here

```
bookrec/
  app.py                       # Streamlit app (the deliverable)
  requirements.txt
  scripts/
    run_cf_bakeoff.py          # CLI: Baseline vs UBCF vs IBCF (Project 2 shell)
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
    llm_rerank.py              # grounded Gemini re-rank (see docs/LLM_RERANK.md)
  docs/
    LLM_RERANK.md              # prompt strategy, grounding, env setup
```

## Vector representations (no ChromaDB)

Book similarity uses **in-memory** vectors in `content_model.py` (TF-IDF by default, or
`sentence-transformers` via `backend="embeddings"`). At ~10k books, full-catalog scoring
is fast enough that a vector database (Chroma, Pinecone, etc.) adds complexity without
improving the graded pipeline. See `docs/LLM_RERANK.md` for when ANN/vector stores matter
at larger scale.

## LLM personalization layer

```bash
pip install google-generativeai
export GEMINI_API_KEY=your_key   # optional; heuristic fallback if unset
```

Details: [`docs/LLM_RERANK.md`](docs/LLM_RERANK.md).

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
