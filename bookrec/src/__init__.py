"""
bookrec - a Goodreads book recommender (Project 2).

Explore the data, build collaborative-filtering recommenders against a
popularity baseline, add an LLM re-ranking layer, and serve it through a
Streamlit app. Reproducible from Books.csv + Ratings.csv.

Modules
-------
sample_data   : synthetic ratings + book metadata for offline development.
data_loader   : load + clean the assignment CSVs (Books.csv, Ratings.csv).
cf_model      : popularity baseline + collaborative filtering (surprise UBCF/IBCF/SVD).
content_model : content-based similarity used for app explanation grounding.
hybrid        : blend CF + content (optional / extra).
evaluate      : RMSE + Precision/Recall@K + NDCG@K + coverage on a hold-out split.
recommend     : top-N orchestration used by the notebook and the app.
llm_rerank    : LLM personalization layer - re-rank CF Top-N by a stated
                preference, with explanations (Gemini; heuristic fallback).
rag_pipeline  : conversation-aware LLM reranking pipeline used by the app chat.
"""

# Explicit exports document the intended public modules for notebooks and the
# Streamlit app; imports still stay lazy because this file does not import them.
__all__ = [
    "sample_data",
    "data_loader",
    "cf_model",
    "content_model",
    "hybrid",
    "evaluate",
    "recommend",
    "llm_rerank",
    "rag_pipeline",
]
