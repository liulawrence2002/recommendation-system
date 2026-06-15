# Goodreads Book Recommender — Project 2

OPAN 6604 recommendation systems project: build and compare collaborative-filtering
models on the Goodreads dataset, then wrap the best model in a **Streamlit app** with
an optional **LLM personalization layer**.

## Quick start

```bash
cd bookrec
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env          # add your GEMINI_API_KEY locally (never commit .env)
streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501).

## Repository layout

```
.
├── README.md                       # this file
├── bookrec/                        # main application (start here)
│   ├── app.py                      # Streamlit deliverable
│   ├── requirements.txt
│   ├── scripts/run_cf_bakeoff.py   # CLI model comparison
│   ├── src/                        # reusable pipeline modules
│   ├── data/                       # Books.csv + Ratings.csv (assignment data)
│   ├── notebooks/                  # project build notebook
│   └── docs/                       # LLM prompt strategy
├── class/                          # Week 3 lecture + exercise (MovieLens)
└── docs/                           # theory reference guides
```

## What the app does

| Tab | Purpose |
|-----|---------|
| **Recommendations** | Pick a user → CF Top-N → optional LLM re-rank by mood/genre |
| **Model evaluation** | Baseline vs UBCF vs IBCF bake-off (RMSE, Precision@10, Recall@10) |

## CLI bake-off

```bash
cd bookrec
python scripts/run_cf_bakeoff.py
```

## Data

The pipeline runs from two files in `bookrec/data/`:

- `Books.csv` — ~9,964 books
- `Ratings.csv` — ~164,728 explicit 1–5 star ratings

A synthetic sample is available in the app for offline testing.

## Team deliverables checklist

- [x] Collaborative filtering models (Baseline, UBCF, IBCF)
- [x] Hold-out evaluation (RMSE + Precision/Recall@10)
- [x] Streamlit app with CF + LLM layer
- [ ] EDA notebook cells + charts
- [ ] Slide deck (PDF, ≤ 8 slides)
- [ ] 2–3 min demo video

See `bookrec/PROJECT_FRAMEWORK.md` for the full rubric map.

## Development notes

- `numpy<2.0` is required for `scikit-surprise` on many platforms.
- If `pip install scikit-surprise` fails: `conda install -c conda-forge scikit-surprise`
- Never commit API keys. Use `bookrec/.env` or `.streamlit/secrets.toml` (both gitignored).



---
### potentially, add filters on decade, genre , potentially book , author ... 
- update the hero landing page text 
- there is a gap between the quality check and the  ready where you are
-   