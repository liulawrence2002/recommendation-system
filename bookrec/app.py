"""
Streamlit app - Goodreads book recommender (Project 2).

Flow required by the assignment:
    1. Pick a user.
    2. Show the collaborative-filtering Top-N (popularity baseline also available).
    3. LLM re-ranking: the user states a preference (mood/genre), and an LLM
       re-ranks the CF candidates and explains each pick.

Run from THIS folder so `from src...` and the data path resolve:
    cd bookrec
    streamlit run app.py

The API key is read from the environment (GEMINI_API_KEY) - never hard-coded.
Without a key the app uses a transparent heuristic re-ranker so it always runs.
"""
from __future__ import annotations

import os
import sys
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import data_loader, recommend, llm_rerank          # noqa: E402
from src.cf_model import PopularityModel                    # noqa: E402

try:
    from src import cf_model
    cf_model._require_surprise()
    HAVE_SURPRISE = True
except Exception:
    HAVE_SURPRISE = False

st.set_page_config(page_title="Book Recommender", page_icon="book", layout="wide")


@st.cache_data
def load_data(source: str):
    if source == "Synthetic sample":
        return data_loader.load_sample()
    return data_loader.load("data")            # real Books.csv + Ratings.csv


@st.cache_resource
def build_model(source: str, kind: str):
    ratings, _ = load_data(source)
    if kind == "popularity" or not HAVE_SURPRISE:
        return PopularityModel().fit(ratings)
    return cf_model.CFModel(kind=kind).fit(ratings)


class ScoreAdapter:
    """Wrap any model so recommend_top_n can call .score(...)."""
    content = None
    def __init__(self, model):
        self.model = model
    def score(self, user_id, user_ratings, candidate_ids):
        return self.model.predict_for_user(user_id, candidate_ids)


st.title("Goodreads Book Recommender")
st.caption("Collaborative filtering + an LLM personalization layer.")

with st.sidebar:
    st.header("Settings")
    source = st.selectbox("Data source", ["Real (data/Books.csv, Ratings.csv)", "Synthetic sample"])
    model_choices = (["svd", "ibcf", "ubcf", "baseline", "popularity"]
                     if HAVE_SURPRISE else ["popularity"])
    cf_kind = st.selectbox("Recommender model", model_choices)
    top_n = st.slider("CF Top-N", 5, 30, 10)
    min_ratings = st.slider("Popularity filter (min ratings/book)", 0, 200, 50)
    st.divider()
    st.subheader("LLM layer")
    have_key = bool(os.environ.get("GEMINI_API_KEY"))
    st.caption("GEMINI_API_KEY found." if have_key
               else "No GEMINI_API_KEY -> heuristic fallback re-ranker.")
    if not HAVE_SURPRISE:
        st.warning("scikit-surprise not installed -> only the popularity baseline is available.")

with st.spinner("Loading data..."):
    ratings, books = load_data(source)

model = build_model(source, cf_kind)
scorer = ScoreAdapter(model)

c1, c2, c3 = st.columns(3)
c1.metric("Users", f"{ratings['user_id'].nunique():,}")
c2.metric("Books", f"{len(books):,}")
c3.metric("Ratings", f"{len(ratings):,}")

st.subheader("1 - Collaborative-filtering Top-N")
uid = st.selectbox("Pick a user", sorted(ratings["user_id"].unique())[:1000])

if "cf_recs" not in st.session_state:
    st.session_state["cf_recs"] = None

if st.button("Get recommendations"):
    with st.spinner("Scoring the catalog..."):
        st.session_state["cf_recs"] = recommend.recommend_top_n(
            uid, scorer, ratings, books, top_n=top_n, min_ratings=min_ratings, explain=False)

if st.session_state["cf_recs"] is not None:
    recs = st.session_state["cf_recs"]
    st.dataframe(recs[["book_id", "title", "authors", "score"]],
                 use_container_width=True, hide_index=True)

    st.subheader("2 - LLM re-ranking & personalization")
    pref = st.text_input("Your preference (mood, genre, vibe...)",
                         "something fast-paced and adventurous")
    k = st.slider("How many personalized picks", 3, min(10, len(recs)), 5)
    if st.button("Re-rank with LLM"):
        cands = llm_rerank.candidates_from_recs(recs, books)
        with st.spinner("Asking the LLM to re-rank..."):
            picks, used_llm = llm_rerank.rerank(cands, pref, top_k=k)
        st.caption("Source: " + ("Gemini LLM" if used_llm
                                 else "heuristic fallback (set GEMINI_API_KEY for the real LLM)"))
        for i, p in enumerate(picks, 1):
            with st.container(border=True):
                st.markdown(f"**{i}. {p.title}** - *{p.authors}*")
                st.write(p.explanation)
