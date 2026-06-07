"""
Streamlit app — Goodreads Book Recommender (Project 2).

Tabs:
  1. Recommendations — pick a user, get CF Top-N, optionally LLM re-rank.
  2. Model evaluation — Baseline vs UBCF vs IBCF bake-off (Project 2 shell).

Run from THIS folder so `from src...` and the data path resolve:
    cd bookrec
    streamlit run app.py

The API key is read from the environment (GEMINI_API_KEY) — never hard-coded.
Without a key the app uses a transparent heuristic re-ranker so it always runs.
"""
from __future__ import annotations

import os
import sys

import streamlit as st
from dotenv import load_dotenv

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_APP_DIR, ".env"))

sys.path.insert(0, _APP_DIR)

from src import data_loader, evaluate, llm_rerank, recommend  # noqa: E402
from src.cf_model import PopularityModel  # noqa: E402

try:
    from src import cf_model

    cf_model._require_surprise()
    HAVE_SURPRISE = True
except Exception:
    HAVE_SURPRISE = False

st.set_page_config(
    page_title="Goodreads Book Recommender",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Defaults aligned with the Project 2 teammate shell
DEFAULT_K = 10
DEFAULT_MIN_RATINGS = 20
RANDOM_STATE = 6604


@st.cache_data
def load_data(source: str):
    if source == "Synthetic sample":
        return data_loader.load_sample()
    return data_loader.load("data")


@st.cache_resource
def build_model(source: str, kind: str, k: int = DEFAULT_K):
    ratings, _ = load_data(source)
    if kind == "popularity" or not HAVE_SURPRISE:
        return PopularityModel().fit(ratings)
    return cf_model.CFModel(kind=kind, k=k).fit(ratings)


@st.cache_data(show_spinner="Evaluating models on hold-out set...")
def run_model_bakeoff(source: str, k: int, test_size: float = 0.1):
    ratings, _ = load_data(source)
    train, test = evaluate.train_test_split_ratings(
        ratings, test_size=test_size, seed=RANDOM_STATE
    )
    models = {
        "Baseline": cf_model.CFModel("baseline").fit(train),
        "UBCF · pearson": cf_model.CFModel("ubcf", k=k).fit(train),
        "IBCF · cosine": cf_model.CFModel("ibcf", k=k).fit(train),
    }
    if not HAVE_SURPRISE:
        pop = PopularityModel().fit(train)
        models = {"Popularity baseline": pop}
    return evaluate.compare_cf_models(train, test, models), len(train), len(test)


class ScoreAdapter:
    """Wrap any model so recommend_top_n can call .score(...)."""

    content = None

    def __init__(self, model):
        self.model = model

    def score(self, user_id, user_ratings, candidate_ids):
        return self.model.predict_for_user(user_id, candidate_ids)


# --- Sidebar (shared) ---
with st.sidebar:
    st.header("Settings")
    source = st.selectbox(
        "Data source",
        ["Real (data/Books.csv, Ratings.csv)", "Synthetic sample"],
    )
    k_neighbors = st.slider("CF neighborhood size (k)", 5, 50, DEFAULT_K)
    model_choices = (
        ["ubcf", "ibcf", "baseline", "svd", "popularity"]
        if HAVE_SURPRISE
        else ["popularity"]
    )
    cf_kind = st.selectbox("Recommender model", model_choices)
    top_n = st.slider("CF Top-N", 5, 30, 10)
    min_ratings = st.slider(
        "Popularity filter (min ratings/book)", 0, 200, DEFAULT_MIN_RATINGS
    )
    st.divider()
    st.subheader("LLM layer")
    have_key = bool(os.environ.get("GEMINI_API_KEY"))
    try:
        import google.generativeai  # noqa: F401

        have_sdk = True
    except Exception:
        have_sdk = False
    if have_key and have_sdk:
        st.caption("GEMINI_API_KEY + SDK ready (Gemini re-rank).")
    elif have_key:
        st.caption("GEMINI_API_KEY set; install: pip install google-generativeai")
    else:
        st.caption("No GEMINI_API_KEY → heuristic fallback re-ranker.")
    with st.expander("How the LLM layer works"):
        st.markdown(
            "1. CF produces Top-N candidates with real `book_id`s only.\n"
            "2. Metadata (title, authors, year, avg rating) is sent to Gemini.\n"
            "3. The model **re-ranks** — it cannot invent books.\n\n"
            f"Model: `{llm_rerank.DEFAULT_MODEL}` · See `docs/LLM_RERANK.md`."
        )
    if not HAVE_SURPRISE:
        st.warning("scikit-surprise not installed → popularity baseline only.")

# --- Header ---
st.title("Goodreads Book Recommender")
st.caption("Project 2 — collaborative filtering + LLM personalization layer.")

with st.spinner("Loading data..."):
    ratings, books = load_data(source)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Users", f"{ratings['user_id'].nunique():,}")
c2.metric("Books", f"{len(books):,}")
c3.metric("Ratings", f"{len(ratings):,}")
sparsity = 1 - len(ratings) / (
    ratings["user_id"].nunique() * ratings["book_id"].nunique()
)
c4.metric("Sparsity", f"{sparsity:.1%}")

tab_recs, tab_eval = st.tabs(["Recommendations", "Model evaluation"])

# --- Tab 1: Recommendations ---
with tab_recs:
    model = build_model(source, cf_kind, k=k_neighbors)
    scorer = ScoreAdapter(model)

    st.subheader("1 — Collaborative-filtering Top-N")
    uid = st.selectbox(
        "Pick a user",
        sorted(ratings["user_id"].unique())[:1000],
        key="rec_user",
    )

    if "cf_recs" not in st.session_state:
        st.session_state["cf_recs"] = None

    if st.button("Get recommendations", type="primary"):
        with st.spinner("Scoring the catalog..."):
            st.session_state["cf_recs"] = recommend.recommend_top_n(
                uid,
                scorer,
                ratings,
                books,
                top_n=top_n,
                min_ratings=min_ratings,
                explain=False,
            )

    if st.session_state["cf_recs"] is not None:
        recs = st.session_state["cf_recs"]
        st.dataframe(
            recs[["book_id", "title", "authors", "score"]],
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("2 — LLM re-ranking & personalization")
        pref = st.text_input(
            "Your preference (mood, genre, vibe...)",
            "something fast-paced and adventurous",
        )
        k = st.slider("How many personalized picks", 3, min(10, len(recs)), 5)
        if st.button("Re-rank with LLM"):
            cands = llm_rerank.candidates_from_recs(recs, books)
            with st.spinner("Asking the LLM to re-rank..."):
                picks, used_llm = llm_rerank.rerank(cands, pref, top_k=k)
            src = "Gemini LLM" if used_llm else "heuristic fallback"
            if not used_llm and have_key and not have_sdk:
                src += " (install google-generativeai)"
            elif not used_llm and not have_key:
                src += " (set GEMINI_API_KEY)"
            st.caption(f"Source: {src} · model: {llm_rerank.DEFAULT_MODEL}")
            for i, p in enumerate(picks, 1):
                with st.container(border=True):
                    st.markdown(f"**{i}. {p.title}** — *{p.authors}*")
                    st.write(p.explanation)

# --- Tab 2: Model evaluation (Project 2 bake-off) ---
with tab_eval:
    st.subheader("CF model bake-off")
    st.markdown(
        "Compare **Baseline**, **UBCF (Pearson)**, and **IBCF (cosine)** on a "
        f"90/10 hold-out split (`random_state={RANDOM_STATE}`). "
        "A book counts as *relevant* if its true rating ≥ 4.0."
    )

    if st.button("Run evaluation", type="primary", key="run_eval"):
        st.session_state["eval_ran"] = True

    if st.session_state.get("eval_ran") and HAVE_SURPRISE:
        results, n_train, n_test = run_model_bakeoff(source, k_neighbors)
        st.caption(f"Train: {n_train:,} ratings · Test: {n_test:,} ratings")
        st.dataframe(results, use_container_width=True)

        best_rmse = results["RMSE"].idxmin()
        best_prec = results["Precision@10"].idxmax()
        st.info(
            f"Lowest RMSE: **{best_rmse}** ({results.loc[best_rmse, 'RMSE']:.4f}). "
            f"Best Precision@10: **{best_prec}** "
            f"({results.loc[best_prec, 'Precision@10']:.4f}). "
            "For a top-10 list, ranking metrics often matter more than RMSE."
        )

        with st.expander("Deploy recommendation"):
            deploy_model = st.selectbox(
                "Model to use in Recommendations tab",
                ["ubcf", "ibcf", "baseline"],
            )
            st.caption(
                f"Switch to the Recommendations tab and select **{deploy_model}** "
                "in the sidebar to deploy this model."
            )
    elif st.session_state.get("eval_ran"):
        st.warning("Install scikit-surprise to run the CF bake-off.")
