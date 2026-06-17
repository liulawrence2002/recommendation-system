# Submission Checklist

This file labels the pieces needed to run `bookrec/app.py` as it is currently
written.

## Run command

```powershell
cd bookrec
pip install -r requirements.txt
streamlit run app.py
```

For live Gemini reranking, set `GEMINI_API_KEY` in your local environment or in
Streamlit Community Cloud secrets. Do not submit `.env`; the app still runs with
the transparent heuristic fallback when no key is present.

## Required app bundle

Submit these files/folders for the Streamlit app:

```text
bookrec/
  app.py
  requirements.txt
  .streamlit/config.toml
  assets/
    book.webp
    nodes.webp
    spiral.webp
  data/
    Books.csv
    Ratings.csv
    README.md
  src/
    __init__.py
    data_loader.py
    sample_data.py
    cf_model.py
    content_model.py
    recommend.py
    llm_rerank.py
    rag_pipeline.py
```

Why these matter:

- `data_loader.py` reads `Books.csv` and `Ratings.csv` into the shared schema.
- `cf_model.py` trains the selected user-based collaborative filter and provides
  the popularity fallback if `scikit-surprise` is unavailable.
- `recommend.py` builds the Top-N candidate list that the app displays.
- `content_model.py` supplies "similar to a book you liked" grounding for the
  reranker explanations.
- `llm_rerank.py` and `rag_pipeline.py` power the Gemini/heuristic chat
  reranking flow.
- `sample_data.py` is not used by the default real-data path, but it is small
  and supports the loader's synthetic-data fallback.
- `assets/` is not required for logic, but it preserves the app's intended
  visual design.

## Modeling evidence to submit with the project

These are not required for `app.py` to run, but they support the assignment's
coding and analysis requirements:

```text
bookrec/notebooks/01_build_book_recommender.ipynb
bookrec/notebooks/02_eda_deep_dive.ipynb
bookrec/scripts/run_cf_bakeoff.py
bookrec/src/evaluate.py
bookrec/src/hybrid.py
bookrec/reports/
bookrec/docs/
BookRec_Project2_Deck.pdf
```

`evaluate.py` is the RMSE/Precision/Recall/NDCG evidence path. `hybrid.py` is a
notebook extension and is not part of the final Streamlit runtime.

## Do not submit

Exclude local or sensitive files:

```text
bookrec/.env
bookrec/.venv/
bookrec/__pycache__/
bookrec/src/__pycache__/
bookrec/streamlit.log
bookrec/_st_run.log
```

Also exclude any API key from slides, screenshots, notebooks, or commit history.
