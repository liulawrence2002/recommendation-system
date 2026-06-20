"""
Streamlit app - Goodreads Book Recommender (Project 2).

Run from this folder so imports and data paths resolve:
    cd bookrec
    streamlit run app.py

The API key is read from the environment (GEMINI_API_KEY). The chat re-ranker is
LLM-only (Gemini) — without a key the chat is disabled and shows a clear notice.
"""
from __future__ import annotations

import base64
from dataclasses import asdict
from html import escape
import os
import re
import sys
import urllib.parse
import warnings

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# ======================================================================
# Inlined source (formerly bookrec/src/*). Kept in src/ too for the
# archived analysis code; this copy makes app.py runnable standalone.
# ======================================================================

# ===== inlined from src/data_loader.py =====
import os
import pandas as pd


## fix coding , part of preprossing and cleaning the data for use in the pipeline (helper function)
def _fix_mojibake(s):
    """Repair UTF-8 text mis-decoded as latin-1 (e.g. 'GrandPrÃ©' -> 'GrandPré')."""
    if not isinstance(s, str):
        return s
    try:
        return s.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s


def load(data_dir: str = "data", fix_encoding: bool = True):
    """Primary loader: read the assignment Books.csv + Ratings.csv.

    Reproducible from just those two files. We look for them in ``data_dir``
    first, then fall back to the project root (data_dir's parent/grandparent) and
    the current working directory — so the CSVs work whether they live in
    ``bookrec/data/`` or are dropped in the repo root. There is no shelf-tags
    file in this dataset, so an empty `tags` column is added. Returns
    (ratings, books) in the package contract.
    """
    # Candidate folders to search, in priority order (deduped, keeping order).
    _candidates = []
    for d in (data_dir,
              os.path.dirname(data_dir),                 # e.g. bookrec/
              os.path.dirname(os.path.dirname(data_dir)),  # e.g. repo root
              os.getcwd()):
        if d and d not in _candidates:
            _candidates.append(d)

    def _path(name):
        for d in _candidates:
            p = os.path.join(d, name)
            if os.path.exists(p):
                return p
        searched = ", ".join(repr(d) for d in _candidates)
        raise FileNotFoundError(
            f"Could not find {name}. Looked in: {searched}. "
            f"Place Books.csv and Ratings.csv in one of these (e.g. bookrec/data/ "
            f"or the repo root)."
        )

    books = pd.read_csv(_path("Books.csv"))
    ratings = pd.read_csv(_path("Ratings.csv"))

    if fix_encoding:
        for col in ("authors", "title"):
            if col in books.columns:
                # The assignment CSV has a few UTF-8 strings that were decoded
                # as latin-1; repair display text before any UI/model use.
                books[col] = books[col].map(_fix_mojibake)

    books["tags"] = ""                       # no shelf tags in this dataset

    books = _coerce_books(books)
    ratings = ratings[["user_id", "book_id", "rating"]].copy()
    ratings["rating"] = ratings["rating"].astype(float)
    return ratings, books


def _coerce_books(books: pd.DataFrame) -> pd.DataFrame:
    """Ensure all contract columns exist with safe defaults."""
    defaults = {
        "title": "Untitled", "authors": "Unknown",
        "original_publication_year": 0, "average_rating": 0.0,
        "ratings_count": 0, "tags": "",
    }
    for col, val in defaults.items():
        # Downstream modules select these columns unconditionally, so create
        # absent optional fields before filling missing values.
        if col not in books.columns:
            books[col] = val
        books[col] = books[col].fillna(val)
    return books


# ===== inlined from src/cf_model.py =====
import pandas as pd

RATING_SCALE = (1.0, 5.0)   # this dataset uses 1..5 stars


def _require_surprise():
    # Import Surprise lazily so the module stays importable on machines without
    # the compiled wheel; the app's filtering block stops with a clear message
    # if it's missing (collaborative filtering requires it).
    try:
        import surprise  # noqa: F401
        return surprise
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "scikit-surprise is needed for collaborative filtering.\n"
            "  pip install 'numpy<2.0' && pip install scikit-surprise\n"
            "  (or: conda install -c conda-forge scikit-surprise)\n"
            f"original error: {e}"
        )


def build_dataset(ratings: pd.DataFrame):
    """Wrap a [user_id, book_id, rating] frame as a Surprise Dataset."""
    surprise = _require_surprise()
    reader = surprise.Reader(rating_scale=RATING_SCALE)
    return surprise.Dataset.load_from_df(
        ratings[["user_id", "book_id", "rating"]], reader)


def make_model(kind: str = "ubcf", k: int = 10, sim_name: str | None = None):
    """Factory: 'ubcf' (user-based KNN) or 'ibcf' (item-based KNN).

    sim_name overrides the KNN similarity (e.g. "pearson_baseline"). When None
    the defaults apply: ubcf -> pearson, ibcf -> cosine. The corrected Top-N
    audit picks pearson_baseline; the app ships UBCF and documents IBCF as the
    higher-RAM swap (see the model-selection block below).
    """
    surprise = _require_surprise()
    if kind == "ubcf":
        sim = {"name": sim_name or "pearson", "user_based": True}
        return surprise.KNNBasic(k=k, sim_options=sim, verbose=False)
    if kind == "ibcf":
        sim = {"name": sim_name or "cosine", "user_based": False}
        return surprise.KNNBasic(k=k, sim_options=sim, verbose=False)
    raise ValueError(f"unknown kind: {kind!r} (only 'ubcf' and 'ibcf' are supported)")


class CFModel:
    """Uniform wrapper: train, then .predict(user, book) -> estimated rating."""

    def __init__(self, kind: str = "ubcf", k: int = 10, sim_name: str | None = None):
        self.kind = kind
        self.k = k
        self.sim_name = sim_name
        self.algo = make_model(kind, k, sim_name)
        self._global_mean = 3.5

    def fit(self, ratings: pd.DataFrame, trainset=None):
        if trainset is None:
            trainset = build_dataset(ratings).build_full_trainset()
        # Accepting an external trainset lets evaluation notebooks reuse the
        # exact same split instead of silently rebuilding on all ratings.
        self.algo.fit(trainset)
        self._global_mean = trainset.global_mean
        return self

    def predict(self, user_id, book_id) -> float:
        return self.algo.predict(user_id, book_id).est

    def predict_for_user(self, user_id, book_ids) -> pd.Series:
        est = [self.predict(user_id, b) for b in book_ids]
        return pd.Series(est, index=list(book_ids), name="cf_score")


# ===== inlined from src/content_model.py =====
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


def _content_document(books: pd.DataFrame) -> pd.Series:
    """One text blob per book. Repeat tags so shelves dominate the signal."""
    title = books["title"].fillna("").astype(str)
    authors = books["authors"].fillna("").astype(str)
    tags = books["tags"].fillna("").astype(str)
    return title + " " + authors + " " + tags + " " + tags  # tags weighted x2


class ContentModel:
    """TF-IDF content vectors over book metadata, used only to ground the
    re-ranker's explanations (candidate -> the reader's most similar favorite)."""

    def __init__(self):
        self.books = None
        self.item_vectors = None          # [n_books, d], L2-normalized
        self._row_of = {}                 # book_id -> row index

    def fit(self, books: pd.DataFrame):
        self.books = books.reset_index(drop=True)
        # Row lookup keeps similarity code fast and avoids repeated DataFrame
        # filtering when scoring thousands of candidate ids.
        self._row_of = {bid: i for i, bid in enumerate(self.books["book_id"])}
        docs = _content_document(self.books)
        # Unigrams + bigrams capture titles/authors and common shelf phrases
        # while staying lightweight enough for Streamlit startup.
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), min_df=1, max_df=0.9,
            stop_words="english", sublinear_tf=True)
        X = self.vectorizer.fit_transform(docs)        # sparse, already L2-normalized
        self.item_vectors = normalize(X)               # be explicit
        return self


    # --- grounding: candidate -> the user's own favorite it most resembles --
    def nearest_examples(self, candidate_ids, liked_ids):
        """For each candidate, the single most content-similar book among the
        user's ``liked_ids`` (books they rated highly).

        Returns ``{candidate_id: (liked_book_id, similarity)}``, skipping any
        candidate with no positive match. Used to *ground* the LLM re-ranker's
        explanations in the reader's real history ("in the spirit of X, which you
        rated highly") instead of letting the model guess from the title alone.
        """
        out = {}
        liked = [(b, self._row_of[b]) for b in liked_ids if b in self._row_of]
        if not liked or self.item_vectors is None:
            return out
        lids, lidx = zip(*liked)
        L = self.item_vectors[list(lidx)]
        for b in candidate_ids:
            r = self._row_of.get(b)
            if r is None:
                continue
            v = self.item_vectors[r]
            sims = L @ (v.T if hasattr(v, "T") else v)
            sims = (np.asarray(sims.todense()).ravel()
                    if hasattr(sims, "todense") else np.asarray(sims).ravel())
            order = np.argsort(-sims)
            for j in order:                      # take the best match that isn't itself
                if lids[j] != b and sims[j] > 0:
                    out[b] = (lids[j], float(sims[j]))
                    break
        return out


# ===== inlined from src/llm_rerank.py =====
import json
import os
import re
import time
from dataclasses import dataclass
from enum import Enum

try:  # Pydantic ships as a dependency of the google-genai SDK.
    from pydantic import BaseModel, Field
    _HAVE_PYDANTIC = True
except Exception:  # keep the module importable even without pydantic installed
    _HAVE_PYDANTIC = False

# Provider/model for the LLM layer. gemini-2.5-flash-lite is the model used in
# the Week-4 class exercise — fast, free-tier friendly, and it supports the
# structured-output mode (response_schema) we rely on to force valid JSON instead
# of regex-parsing free text. Cite this pair in the deck:
#     provider = "Google", model = DEFAULT_MODEL
DEFAULT_MODEL = "gemini-2.5-flash-lite"

# Persona + house style passed as the Week-4 `system_instruction` on the
# explanation stages. Giving the model a consistent voice plus an explicit
# "be specific to each book, never repeat yourself" rule is what turns one
# generic reason repeated down the list into a distinct, grounded "why" per pick.
PERSONA = ( "You are BookRec, a careful reader-advisory librarian embedded inside a " 
           "collaborative-filtering book recommender. Your role is to personalize and "
            "re-rank ONLY the candidate books already produced by the recommendation model; "
              "you must never invent, replace, or add books outside the provided candidate list. "
                "Use the available evidence for each candidate — title, author, publication year, " 
                "average rating, collaborative-filtering score, similarity to books the reader "
                  "already liked, relevance notes, and avoid-flags — to decide the final order. " 
                  "Prioritize the reader's stated mood, genre, pace, themes, recency preference, "
                    "and any things they asked to avoid. If a book conflicts with an avoid preference, "
                      "demote it unless there is a clear reason to keep it. Write explanations that are " 
                      "specific to each individual book and grounded in the provided metadata or known " 
                      "context; do not fabricate plot details, themes, or genres when the metadata does "
                        "not support them. If the available metadata is limited, explain the fit using the "
                          "author, era, rating, CF score, or similarity to the reader's favorites. Every "
                            "explanation should be one natural sentence, friendly and concrete, with no repeated "
                              "phrasing across the list. Sound like a smart librarian helping a reader, not a "
                                "marketing blurb or a generic AI assistant." )


@dataclass
class RerankedPick:
    book_id: int
    title: str
    authors: str
    explanation: str          # WHY this book earned its rank — ties to the reader intent
    description: str = ""      # ONE neutral sentence describing what the book IS / is about

# Passing a Pydantic model as Gemini's `response_schema` makes the API return
# JSON that already matches this shape (response.parsed is typed), so we drop the
# brittle regex parsing. This is the exact technique from the Week-4 exercise.
if _HAVE_PYDANTIC:
    class RerankItem(BaseModel):
        book_id: int = Field(description="A book_id taken ONLY from the candidate list.")
        description: str = Field(description="One neutral sentence on what the book is about.")
        explanation: str = Field(description="One sentence on why it earns this rank.")

    RERANK_LIST_SCHEMA = list[RerankItem]
else:  # no pydantic -> structured output is unavailable; callers use the text path
    RerankItem = None
    RERANK_LIST_SCHEMA = None


def _candidate_table(candidates):
    """candidates: list of dicts with book_id, title, authors, year, average_rating,
    cf_score, and optionally similar_to (a book the reader already rated highly).
    Returns a compact, numbered context string for the prompt."""
    lines = []
    for c in candidates:
        line = (
            f"[{c['book_id']}] \"{c['title']}\" by {c.get('authors','?')} "
            f"({c.get('year','?')}), avg_rating={c.get('average_rating','?')}, "
            f"cf_score={c.get('cf_score','?')}"
        )
        if c.get("similar_to"):
            line += f', similar_to_reader_favorite="{c["similar_to"]}"'
        lines.append(line)
    return "\n".join(lines)


def _synth_description(c) -> str:
    """Synthesize a neutral one-sentence book description from the metadata we
    have (author, year, average_rating). Used when no synopsis/genre text exists
    and as the offline fallback. Never crashes on missing/"?" values."""
    title = str(c.get("title", "") or "this book").strip()
    authors = str(c.get("authors", "") or "").strip()
    year = c.get("year", "?")
    avg = c.get("average_rating", "?")

    parts = [f"“{title}”"]
    if authors and authors != "?":
        parts.append(f"by {authors}")
    # only mention the year if it looks like a real value
    if year not in (None, "", "?") and str(year).strip() not in ("", "?"):
        parts.append(f"published in {year}")
    sentence = " ".join(parts)
    # close with a rating clause when available, otherwise a neutral fallback
    if avg not in (None, "", "?") and str(avg).strip() not in ("", "?"):
        sentence += f", a reader-rated book averaging {avg}/5."
    else:
        sentence += ", a book from this collection."
    return sentence


def _have_new_sdk() -> bool:
    """The Week-4 google-genai SDK (preferred — supports response_schema)."""
    try:
        from google import genai  # noqa: F401
        return True
    except Exception:
        return False


def _have_old_sdk() -> bool:
    """The legacy google-generativeai SDK (text-only fallback)."""
    try:
        import google.generativeai  # noqa: F401
        return True
    except Exception:
        return False


def _sdk_available() -> bool:
    return _have_new_sdk() or _have_old_sdk()


def _call_gemini(prompt: str, model: str) -> str:
    """Legacy text call (old google-generativeai SDK). Used only as a fallback
    when the new google-genai SDK isn't installed; structured output is preferred."""
    import google.generativeai as genai
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    resp = genai.GenerativeModel(model).generate_content(prompt)
    return resp.text


def _to_plain(parsed):
    """Convert google-genai `.parsed` (Pydantic models / enums / lists thereof)
    into the plain dict / list-of-dicts shape every stage parser already expects."""
    if parsed is None:
        return None
    if isinstance(parsed, list):
        return [_to_plain(x) for x in parsed]
    if isinstance(parsed, Enum):
        return parsed.value
    if hasattr(parsed, "model_dump"):
        # mode="json" serializes Enum fields (e.g. _Pace/_Recency) to their string
        # values ("any") rather than leaving them as enum members, whose str() is
        # "_Pace.ANY" — that leaked into the UI and falsely tripped the clarify gate.
        return parsed.model_dump(mode="json")
    return parsed


# Small in-memory cache of structured LLM responses. Each chat turn fires up to
# three Gemini calls (intent -> score -> re-rank); identical prompts (e.g. the
# user re-sends a suggestion chip, or Streamlit replays a turn) return instantly
# instead of re-billing the API. Keyed by the full prompt + model + system + schema
# so different inputs never collide; values are deep-copied in and out so callers
# can't mutate the cache. This directly mitigates the cost/latency + rate-limit
# (HTTP 429) challenges called out in the business write-up.
_LLM_CACHE: dict = {}
_LLM_CACHE_MAX = 256

# The free tier's per-minute (RPM) burst limit clears in a second or two, so we
# retry a rate-limited call a couple of times with exponential backoff before
# giving up. On a genuinely exhausted quota (or other hard error) the call raises
# and the chat turn surfaces an "AI temporarily unavailable" message.
_LLM_MAX_RETRIES = 2
_LLM_RETRY_BASE = 1.0  # seconds; backoff is _LLM_RETRY_BASE * 2**attempt -> ~1s, 2s


def _is_rate_limit(exc) -> bool:
    """True if an exception looks like a Gemini 429 / quota error. Robust across
    SDK versions: checks the structured status code and the message text."""
    if getattr(exc, "code", None) == 429:
        return True
    text = str(exc).lower()
    return any(s in text for s in ("429", "resource_exhausted", "rate limit", "quota"))


def _cache_key(prompt: str, schema, model: str, system: str | None):
    schema_name = getattr(schema, "__name__", None) or repr(schema)
    return (model, schema_name, system or "", prompt)


def _structured(prompt: str, schema, model: str = DEFAULT_MODEL, *,
                is_list: bool = False, system: str | None = None):
    """Preferred LLM path — Week-4 structured output.

    Returns plain Python (dict for an object schema, list[dict] for a list
    schema, str for an Enum) so existing ``.get(...)`` parsing keeps working
    unchanged. Prefers the new google-genai SDK with ``response_schema``; if only
    the legacy SDK is present (or pydantic is missing), falls back to a text call
    plus a tolerant regex parse. Results are cached in-process (see ``_LLM_CACHE``).
    Raises on hard failure so the chat turn can surface an error to the reader.
    """
    import copy

    key = _cache_key(prompt, schema, model, system)
    if key in _LLM_CACHE:
        return copy.deepcopy(_LLM_CACHE[key])

    for attempt in range(_LLM_MAX_RETRIES + 1):
        try:
            result = _structured_uncached(
                prompt, schema, model, is_list=is_list, system=system
            )
            break
        except Exception as exc:
            if attempt < _LLM_MAX_RETRIES and _is_rate_limit(exc):
                time.sleep(_LLM_RETRY_BASE * (2 ** attempt))
                continue
            raise

    if len(_LLM_CACHE) < _LLM_CACHE_MAX:
        _LLM_CACHE[key] = copy.deepcopy(result)
    return result


def _structured_uncached(prompt: str, schema, model: str = DEFAULT_MODEL, *,
                         is_list: bool = False, system: str | None = None):
    if schema is not None and _have_new_sdk():
        from google import genai
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        is_enum = isinstance(schema, type) and issubclass(schema, Enum)
        config = {
            "response_mime_type": "text/x.enum" if is_enum else "application/json",
            "response_schema": schema,
        }
        if system:
            config["system_instruction"] = system
        resp = client.models.generate_content(model=model, contents=prompt, config=config)
        plain = _to_plain(resp.parsed)
        if plain is None or (is_list and not plain):
            raise ValueError("empty structured response")
        return plain

    # Fallback: legacy SDK text + tolerant regex parse.
    raw = _call_gemini(prompt, model)
    if is_list:
        return _parse_json_list(raw)
    if isinstance(schema, type) and issubclass(schema, Enum):
        return raw.strip()
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        raise ValueError("no JSON object in response")
    return json.loads(m.group(0))


def _parse_json_list(text: str):
    """Pull the first JSON array out of the model's reply (legacy text path)."""
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return []


def candidates_from_recs(recs_df, books):
    """Helper: turn a recommend_top_n() DataFrame into the candidate dicts this
    module expects, pulling metadata from the books frame."""
    meta = books.set_index("book_id")
    out = []
    for r in recs_df.itertuples(index=False):
        # Start with placeholders because many CSV fields are optional or NaN.
        year, avg = "?", "?"
        if r.book_id in meta.index:
            row = meta.loc[r.book_id]
            y = row.get("original_publication_year", None)
            if y is not None and y == y:  # not NaN
                year = int(float(y))
            a = row.get("average_rating", None)
            if a is not None and a == a:
                avg = round(float(a), 2)
        out.append({
            "book_id": r.book_id,
            "title": getattr(r, "title", ""),
            "authors": getattr(r, "authors", ""),
            "year": year,
            "average_rating": avg,
            "cf_score": round(float(getattr(r, "score", 0.0)), 3),
        })
    return out


# rag_pipeline referenced its sibling as ``llm_rerank.<fn>``; both are inlined
# into this file now, so point that name at this module to keep those calls
# resolving (covers conditionally-defined names like RERANK_LIST_SCHEMA too).
llm_rerank = sys.modules[__name__]


# ===== inlined from src/rag_pipeline.py =====
import json
import os
import re
from dataclasses import dataclass, field


# --- Week-4 structured-output schemas for each DAG stage -----------------------
# Each LLM stage forces Gemini to return JSON matching one of these schemas
# (response.parsed is typed), instead of parsing free text. `pace` and `recency`
# use Enum fields — the Week-4 "enum mode" technique applied to constrained
# facets, so the model can only emit a value the downstream scorer understands.
if llm_rerank._HAVE_PYDANTIC:
    from enum import Enum

    from pydantic import BaseModel, Field

    class _Pace(str, Enum):
        FAST = "fast"
        SLOW = "slow"
        ANY = "any"

    class _Recency(str, Enum):
        RECENT = "recent"
        CLASSIC = "classic"
        ANY = "any"

    class IntentSchema(BaseModel):
        mood: str = Field(description="One-word mood, or empty string if unclear.")
        genres: list[str] = Field(description="Genres the reader wants.")
        themes: list[str] = Field(description="Topics/themes the reader wants.")
        pace: _Pace = Field(description="Desired pacing.")
        avoid: list[str] = Field(description="Things to steer away from.")
        recency: _Recency = Field(description="Preferred era.")
        summary: str = Field(description="One sentence restating the whole request.")

    class ScoreItem(BaseModel):
        book_id: int = Field(description="A book_id taken ONLY from the candidate list.")
        relevance: float = Field(description="Match score in [0,1].")
        reason: str = Field(description="One clause tying the book to the intent.")
        flags: list[str] = Field(description='intent.avoid hits, e.g. "avoid:romance".')

    class ClarifySchema(BaseModel):
        question: str = Field(description="One short, friendly clarifying question.")
        options: list[str] = Field(description="3-4 concise tap-to-answer options.")

    SCORE_LIST_SCHEMA = list[ScoreItem]
else:  # no pydantic -> stages fall back to the legacy text + regex path
    IntentSchema = ScoreItem = ClarifySchema = None
    SCORE_LIST_SCHEMA = None

# --- dataclasses ---------------------------------------------------------------

@dataclass
class Intent:
    """Stage A output: a structured reading of the whole conversation."""

    mood: str = ""
    genres: list = field(default_factory=list)
    themes: list = field(default_factory=list)
    pace: str = "any"          # "fast" | "slow" | "any"
    avoid: list = field(default_factory=list)
    recency: str = "any"       # "recent" | "classic" | "any"
    summary: str = ""          # one-line restatement of the whole thread


@dataclass
class ScoredCandidate:
    """Stage B output: one relevance judgement per candidate book."""

    book_id: int
    relevance: float = 0.0     # 0..1
    reason: str = ""
    flags: list = field(default_factory=list)   # e.g. ["avoid:romance"]


@dataclass
class ClarifyingQuestion:
    """Clarify-gate output: a question to ask before ranking a vague request."""

    prompt: str = ""           # the question text shown to the reader
    options: list = field(default_factory=list)   # 3-4 short quick-reply strings
    reason: str = ""           # why we're asking (surfaced in the trace)


@dataclass
class StageTrace:
    """A single DAG node, surfaced in the UI reasoning trace."""

    name: str                  # "Intent" | "Clarify" | "Scoring" | "Re-rank"
    used_llm: bool
    output: dict               # JSON-safe summary of the stage payload
    note: str = ""


@dataclass
class PipelineResult:
    picks: list                # list[RerankedPick], final ordered shortlist
    intent: Intent
    scored: list               # list[ScoredCandidate]
    used_llm: bool             # True if ANY stage used the live LLM
    trace: list                # list[StageTrace] in DAG order
    source: str                # display label, mirrors llm_rerank convention
    question: object = None    # ClarifyingQuestion when the DAG paused to ask;
    #                            None when it ran through to picks


# --- shared helpers ------------------------------------------------------------

def _describe_book(c) -> str:
    """Synthesize a neutral one-sentence description of a candidate book from its
    metadata (title/authors/year/average_rating). There is no synopsis/genre text
    in the candidate dicts, so this is what feeds the description in the fallback
    path and backfills any blank description from the live LLM. Delegates to
    ``llm_rerank._synth_description`` so both modules describe books identically;
    never crashes on missing/"?" values."""
    return llm_rerank._synth_description(c or {})


def _compose_summary(user_turns) -> str:
    """Fold the conversation into one preference string.

    The first turn is the base request; later turns are refinements applied in
    order (newest last). Ported from the app's old ``compose_preference`` so the
    summary stays conversation-aware even when Stage A falls back.
    """
    turns = [t.strip() for t in user_turns if t and t.strip()]
    if not turns:
        return ""
    base = turns[0]
    if len(turns) == 1:
        return base
    refinements = "; ".join(turns[1:])
    return (
        f"{base}. The reader then refined the request (apply in order, newest "
        f"last): {refinements}. Keep the original intent but prioritize the most "
        f"recent refinement."
    )


def _as_str_list(value) -> list:
    """Coerce a model-supplied field into a clean list of short strings."""
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    out = []
    for item in value:
        text = str(item).strip()
        if text:
            out.append(text)
    return out


# --- Stage A: intent extraction ------------------------------------------------

def _intent_prompt(user_turns) -> str:
    base = user_turns[0].strip()
    refinements = [t.strip() for t in user_turns[1:] if t.strip()]
    refine_block = (
        "\n".join(f"  - {r}" for r in refinements) if refinements else "  (none yet)"
    )
    return (
        "You are the intent-extraction stage of a book recommendation pipeline. "
        "Read the reader's conversation and distill it into structured preferences. "
        "Refinements are applied in order; the MOST RECENT refinement wins on "
        "conflicts, but keep the original intent otherwise.\n\n"
        f'Base request: "{base}"\n'
        f"Refinements (oldest first):\n{refine_block}\n\n"
        "Respond with STRICT JSON, exactly this shape and nothing else:\n"
        '{"mood": "<one word or empty>", '
        '"genres": ["..."], "themes": ["..."], '
        '"pace": "fast|slow|any", '
        '"avoid": ["things to steer away from"], '
        '"recency": "recent|classic|any", '
        '"summary": "<one sentence restating the full request>"}'
    )


def extract_intent(user_turns, *, model: str = DEFAULT_MODEL):
    """Stage A (LLM-only). Returns ``(Intent, True)``. Raises if the LLM call fails."""
    turns = [t for t in user_turns if t and t.strip()]
    if not turns:
        return Intent(), True

    data = llm_rerank._structured(_intent_prompt(turns), IntentSchema, model) or {}

    summary = str(data.get("summary", "")).strip() or _compose_summary(turns)
    intent = Intent(
        mood=str(data.get("mood", "")).strip(),
        genres=_as_str_list(data.get("genres")),
        themes=_as_str_list(data.get("themes")),
        pace=(str(data.get("pace", "any")).strip().lower() or "any"),
        avoid=_as_str_list(data.get("avoid")),
        recency=(str(data.get("recency", "any")).strip().lower() or "any"),
        summary=summary,
    )
    return intent, True


# --- Clarify gate: ask before ranking when the request is too vague ------------

# Below this many distinct intent signals, the request is "thin" and the gate
# asks one clarifying question instead of ranking. Tunable in one place.
CLARIFY_MIN_SIGNALS = 2
MAX_CLARIFY_ROUNDS = 2


def _intent_specificity(intent: Intent) -> int:
    """Count the distinct, meaningful signals the reader has given us so far.

    Only the structured Intent facets are inspected — the same fields the
    downstream scorer relies on — so the gate's view matches what ranking can
    actually use.
    """
    score = 0
    if intent.mood:
        score += 1
    score += len(intent.genres)
    if intent.pace and intent.pace != "any":
        score += 1
    if intent.recency and intent.recency != "any":
        score += 1
    # Themes are weak signals — the offline fallback fills them with leftover
    # tokens ("good", "recommend"), so a noisy list shouldn't read as specific.
    # Count their presence as at most one signal.
    score += min(len(intent.themes), 1)
    return score


def needs_clarification(intent: Intent, rounds_asked: int,
                        max_rounds: int = MAX_CLARIFY_ROUNDS) -> bool:
    """True when the request is too thin to rank well AND we still have a
    clarifying round left. After ``max_rounds`` we always proceed to ranking."""
    if rounds_asked >= max_rounds:
        return False
    return _intent_specificity(intent) < CLARIFY_MIN_SIGNALS


def _clarify_prompt(intent: Intent) -> str:
    return (
        "You are the clarifying-question stage of a book recommendation pipeline. "
        "The reader's request so far is too vague to rank confidently. Ask ONE "
        "short, friendly question that would most improve the recommendations, and "
        "offer 3-4 concise tap-to-answer options.\n\n"
        f'Reader summary: "{intent.summary}"\n'
        f"Structured intent so far: {_intent_block(intent)}\n\n"
        "Ask about the single most useful MISSING facet (genre, tone/mood, pace, or "
        "recency). Keep options to a few words each.\n"
        "Respond with STRICT JSON, exactly this shape and nothing else:\n"
        '{"question": "<one short question>", '
        '"options": ["<opt>", "<opt>", "<opt>"]}'
    )


def build_clarifying_question(intent: Intent, *, model: str = DEFAULT_MODEL):
    """Clarify gate (LLM-only). Returns ``(ClarifyingQuestion, True)``. Raises on LLM failure."""
    data = llm_rerank._structured(_clarify_prompt(intent), ClarifySchema, model) or {}
    prompt = str(data.get("question", "")).strip()
    options = _as_str_list(data.get("options"))
    if not prompt or not options:
        raise RuntimeError("clarify stage returned no usable question")
    return (
        ClarifyingQuestion(
            prompt=prompt,
            options=options[:4],
            reason="live LLM gate: request was too vague to rank",
        ),
        True,
    )


# --- Stage B: candidate scoring ------------------------------------------------

def _intent_block(intent: Intent) -> str:
    return json.dumps(
        {
            "mood": intent.mood,
            "genres": intent.genres,
            "themes": intent.themes,
            "pace": intent.pace,
            "avoid": intent.avoid,
            "recency": intent.recency,
        },
        ensure_ascii=False,
    )


def _score_prompt(candidates, intent: Intent) -> str:
    table = llm_rerank._candidate_table(candidates)
    ids = [c["book_id"] for c in candidates]
    return (
        "You are the candidate-scoring stage of a book recommendation pipeline. "
        "Score how well EACH candidate matches the reader's structured intent. "
        "Do not drop candidates and do not invent books.\n\n"
        f'Reader summary: "{intent.summary}"\n'
        f"Structured intent: {_intent_block(intent)}\n\n"
        f"Candidates (choose ONLY from these book_ids: {ids}):\n{table}\n\n"
        "Some candidates list similar_to_reader_favorite — a book this reader "
        "already rated highly; treat that as a strong positive signal and you may "
        "cite it in the reason.\n"
        "For every candidate, return a relevance score in [0,1], a one-clause "
        "reason tied to the intent, and a list of flags for any intent.avoid "
        'terms it triggers (e.g. "avoid:romance"; empty list if none).\n'
        "Respond with STRICT JSON: a list of objects "
        '{"book_id": <int>, "relevance": <float 0..1>, "reason": "<str>", '
        '"flags": ["..."]} and nothing else.'
    )


def score_candidates(candidates, intent: Intent, *, model: str = DEFAULT_MODEL):
    """Stage B (LLM-only). Returns ``(list[ScoredCandidate], True)``. Raises on LLM failure."""
    if not candidates:
        return [], True

    by_id = {c["book_id"]: c for c in candidates}

    parsed = llm_rerank._structured(
        _score_prompt(candidates, intent), SCORE_LIST_SCHEMA, model,
        is_list=True, system=llm_rerank.PERSONA,
    ) or []

    scored = []
    seen = set()
    for item in parsed:
        bid = item.get("book_id")
        if bid not in by_id or bid in seen:        # enforce: only real candidates
            continue
        seen.add(bid)
        try:
            relevance = max(0.0, min(1.0, float(item.get("relevance", 0.0))))
        except (TypeError, ValueError):
            relevance = 0.0
        scored.append(
            ScoredCandidate(
                book_id=bid,
                relevance=round(relevance, 3),
                reason=str(item.get("reason", "")).strip(),
                flags=_as_str_list(item.get("flags")),
            )
        )

    if not scored:                                 # nothing usable from the model
        raise RuntimeError("scoring stage returned no usable scores")

    # backfill any candidate the model skipped with a neutral 0 so Stage C still
    # sees every book (they simply rank last).
    if len(scored) < len(candidates):
        for c in candidates:
            if c["book_id"] not in seen:
                scored.append(ScoredCandidate(book_id=c["book_id"], relevance=0.0,
                                              reason="", flags=[]))
    return scored, True


# --- Stage C: final re-rank ----------------------------------------------------

def _rerank_prompt(candidates, scored, intent: Intent, top_k: int) -> str:
    by_id = {c["book_id"]: c for c in candidates}
    lines = []
    for s in scored:
        c = by_id.get(s.book_id, {})
        flag_note = f", flags={s.flags}" if s.flags else ""
        sim_note = (f', similar_to_reader_favorite="{c["similar_to"]}"'
                    if c.get("similar_to") else "")
        lines.append(
            f"[{s.book_id}] \"{c.get('title', '?')}\" by {c.get('authors', '?')} "
            f"— relevance={s.relevance}, note: {s.reason}{flag_note}{sim_note}"
        )
    scored_block = "\n".join(lines)
    ids = [s.book_id for s in scored]
    return (
        "You are the final re-ranking stage of a book recommendation pipeline. "
        "Using the upstream relevance scores and the reader's intent, choose and "
        "ORDER the best matches. Each explanation MUST reference the reader's "
        "intent so the recommendation ties back to the whole conversation.\n\n"
        f'Reader summary: "{intent.summary}"\n'
        f"Structured intent: {_intent_block(intent)}\n\n"
        f"Scored candidates:\n{scored_block}\n\n"
        f"Instructions:\n"
        f"- Select and order the {top_k} best matches.\n"
        f"- Prefer higher relevance; demote anything flagged against intent.avoid.\n"
        f"- Use ONLY these book_ids: {ids}. Do NOT invent books.\n"
        "- For each pick, give a neutral one-sentence DESCRIPTION of what the book "
        "itself is about, plus a one-sentence EXPLANATION of why it earns this rank "
        "given the reader's intent.\n"
        "- Make every EXPLANATION distinct and specific to that book — reference its "
        "own story, tone, author, era, or rating, and how it compares to the other "
        "picks. Never reuse the same reason or wording across picks.\n"
        "- Respond with STRICT JSON: a list of objects "
        '{"book_id": <int>, "description": "<one sentence about the book>", '
        '"explanation": "<one sentence on why it earns this rank>"} and nothing else.'
    )


def _series_key(title: str) -> str:
    """Series name from a title like 'X (The Stormlight Archive, #2)' -> the
    series; '' when the title names no series."""
    m = re.search(r"\(([^,)]+)", title or "")
    return m.group(1).strip().lower() if m else ""


def _author_key(authors: str) -> str:
    """First (primary) author, lower-cased, for the diversity cap."""
    return (authors or "").split(",")[0].strip().lower()


def _diversify(picks, top_k: int, max_per_author: int = 2):
    """Trim an over-long ranked list to ``top_k`` while avoiding monotony:
    at most ``max_per_author`` books per author and one book per series. Demoted
    picks are kept as backfill so we always return up to ``top_k`` — diversity
    never costs us results."""
    kept, overflow = [], []
    seen_authors: dict = {}
    seen_series: set = set()
    for p in picks:
        # Apply diversity after ranking so relevance remains the primary signal;
        # repeated author/series picks are demoted, not discarded.
        a, s = _author_key(p.authors), _series_key(p.title)
        if seen_authors.get(a, 0) >= max_per_author or (s and s in seen_series):
            overflow.append(p)
            continue
        kept.append(p)
        seen_authors[a] = seen_authors.get(a, 0) + 1
        if s:
            seen_series.add(s)
        if len(kept) >= top_k:
            break
    for p in overflow:                              # backfill if caps left us short
        if len(kept) >= top_k:
            break
        kept.append(p)
    return kept[:top_k]


def rerank_with_intent(candidates, scored, intent: Intent, top_k: int = 5, *,
                       model: str = DEFAULT_MODEL):
    """Stage C (LLM-only). Returns ``(list[RerankedPick], True)``. Raises on LLM failure."""
    if not scored:
        return [], True

    by_id = {c["book_id"]: c for c in candidates}

    parsed = llm_rerank._structured(
        _rerank_prompt(candidates, scored, intent, top_k),
        llm_rerank.RERANK_LIST_SCHEMA, model, is_list=True,
        system=llm_rerank.PERSONA,
    ) or []

    picks = []
    seen = set()
    for item in parsed:
        bid = item.get("book_id")
        if bid not in by_id or bid in seen:        # enforce: only real candidates
            continue
        seen.add(bid)
        c = by_id[bid]
        # use the model's description when present; otherwise synthesize from metadata
        desc = str(item.get("description", "")).strip() or _describe_book(c)
        picks.append(
            RerankedPick(
                book_id=bid,
                title=c.get("title", ""),
                authors=c.get("authors", ""),
                explanation=str(item.get("explanation", "")).strip(),
                description=desc,
            )
        )
        if len(picks) >= top_k:
            break

    if not picks:                                  # model returned nothing usable
        raise RuntimeError("re-rank stage returned no usable picks")
    return picks, True


# --- orchestrator --------------------------------------------------------------


def run_pipeline(user_turns, candidates, top_k: int = 5, *,
                 model: str = DEFAULT_MODEL,
                 rounds_asked: int = 0,
                 skip_clarify: bool = False,
                 max_clarify_rounds: int = MAX_CLARIFY_ROUNDS) -> PipelineResult:
    """Run the sequential DAG: extract_intent → [clarify?] → score → rerank.

    Parameters
    ----------
    user_turns : ordered list of the reader's chat messages (strings). The first
        is the base request; later turns are refinements.
    candidates : the CF Top-N as dicts (see ``llm_rerank.candidates_from_recs``).
    top_k : how many final picks to return.
    rounds_asked : how many clarifying questions have already been asked this
        conversation. The gate stops asking once this reaches ``max_clarify_rounds``.
    skip_clarify : when True, bypass the gate entirely and rank immediately (the
        "Just recommend something" escape hatch).

    Returns a :class:`PipelineResult`. When the request is too vague the result
    carries a ``question`` (and empty picks); otherwise it carries the ranked
    picks. Either way a per-stage trace is included. LLM-only: if a stage's
    Gemini call fails, this raises and the caller surfaces an error turn.
    """
    # Stage A
    # Every run starts by converting the free-form chat history into structured
    # facets; later stages only consume this structured contract.
    intent, used_a = extract_intent(user_turns, model=model)
    trace = [
        StageTrace(
            name="Intent",
            used_llm=used_a,
            output={
                "mood": intent.mood,
                "genres": intent.genres,
                "themes": intent.themes,
                "pace": intent.pace,
                "avoid": intent.avoid,
                "recency": intent.recency,
                "summary": intent.summary,
            },
            note="We read your message and pulled out the details that matter.",
        )
    ]

    # Clarify gate: if the request is too thin, ask one question and stop here.
    specificity = _intent_specificity(intent)
    should_ask = not skip_clarify and needs_clarification(
        intent, rounds_asked, max_clarify_rounds
    )
    if should_ask:
        question, used_gate = build_clarifying_question(intent, model=model)
        trace.append(
            StageTrace(
                name="Clarify",
                used_llm=used_gate,
                output={
                    "specificity": specificity,
                    "threshold": CLARIFY_MIN_SIGNALS,
                    "round": rounds_asked + 1,
                    "max_rounds": max_clarify_rounds,
                    "question": question.prompt,
                    "options": question.options,
                },
                note=(
                    "Your request was still a little open-ended, so we asked a "
                    "quick question to narrow things down before recommending "
                    f"(question {rounds_asked + 1} of up to {max_clarify_rounds})."
                ),
            )
        )
        return PipelineResult(
            picks=[],
            intent=intent,
            scored=[],
            used_llm=True,
            trace=trace,
            source=f"Gemini · {model} · paused for clarification",
            question=question,
        )

    # Gate passed — record why we're proceeding straight to ranking.
    if skip_clarify:
        gate_note = "You asked to skip ahead, so we went straight to picking books."
    elif rounds_asked >= max_clarify_rounds:
        gate_note = (
            "We'd already asked a couple of questions, so we went ahead and "
            "recommended with what we knew."
        )
    else:
        gate_note = (
            "Your request was clear enough to act on, so we went straight to "
            "picking books — no need to ask anything."
        )
    trace.append(
        StageTrace(
            name="Clarify",
            used_llm=False,
            output={
                "specificity": specificity,
                "threshold": CLARIFY_MIN_SIGNALS,
                "asked": False,
                "skipped": bool(skip_clarify),
            },
            note=gate_note,
        )
    )

    # Stage B
    scored, used_b = score_candidates(candidates, intent, model=model)
    title_by_id = {c["book_id"]: c.get("title", "") for c in candidates}
    top_scored = sorted(scored, key=lambda s: s.relevance, reverse=True)[:6]
    trace.append(
        StageTrace(
            name="Scoring",
            used_llm=used_b,
            output={
                "scored": [
                    {
                        "book_id": s.book_id,
                        "title": title_by_id.get(s.book_id, ""),
                        "relevance": s.relevance,
                        "reason": s.reason,
                        "flags": s.flags,
                    }
                    for s in top_scored
                ],
                "count": len(scored),
            },
            note=f"We rated all {len(scored)} books on how well they fit what you "
                 f"asked for. Here are the strongest matches.",
        )
    )

    # Stage C — fetch a few extra picks so the diversity pass has room to drop
    # same-author / same-series clusters without falling short of top_k.
    fetch_k = min(len(candidates), top_k + 3)
    picks, used_c = rerank_with_intent(
        candidates, scored, intent, top_k=fetch_k, model=model,
    )
    picks = _diversify(picks, top_k)
    trace.append(
        StageTrace(
            name="Re-rank",
            used_llm=used_c,
            output={
                "picks": [
                    {"book_id": p.book_id, "title": p.title,
                     "description": p.description, "explanation": p.explanation}
                    for p in picks
                ]
            },
            note="We put the best matches in order and wrote a short reason for "
                 "each one.",
        )
    )

    return PipelineResult(
        picks=picks,
        intent=intent,
        scored=scored,
        used_llm=True,
        trace=trace,
        source=f"Gemini · {model} · 3-stage DAG",
    )


# ===== inlined from src/recommend.py =====
import pandas as pd


def popular_books(ratings: pd.DataFrame, min_ratings: int = 20) -> set:
    # Popularity support filter keeps the first-pass CF list away from books
    # whose scores are based on too little observed feedback.
    counts = ratings["book_id"].value_counts()
    return set(counts[counts >= min_ratings].index)


def recommend_top_n(user_id, hybrid, ratings: pd.DataFrame, books: pd.DataFrame,
                    top_n: int = 10, min_ratings: int = 20,
                    allowed_book_ids=None) -> pd.DataFrame:
    """Return a DataFrame of the user's top-N recommended books.

    allowed_book_ids: optional set of book_ids to restrict candidates to (e.g. an
    author/decade/genre filter). When None (default) no restriction is applied and
    behavior is identical to before. When an empty set, no candidates remain.
    """
    seen = set(ratings.loc[ratings["user_id"] == user_id, "book_id"])
    pop = popular_books(ratings, min_ratings)
    # Candidate universe: unseen books, narrowed by the UI filter FIRST so the
    # popularity floor is applied within the user's selection — not the whole
    # catalog. Applying the filter last (as before) let a narrow author/decade
    # pick collapse to zero whenever none of its books cleared the global
    # popularity floor, which looked like the filter "not working".
    unseen = [b for b in books["book_id"] if b not in seen]
    if allowed_book_ids is not None:
        unseen = [b for b in unseen if b in allowed_book_ids]
    candidates = [b for b in unseen if b in pop]
    if not candidates:                          # nothing popular enough — relax
        candidates = unseen                     # the floor rather than return []

    user_ratings = ratings[ratings["user_id"] == user_id]
    scores = hybrid.score(user_id, user_ratings, candidates)
    top = scores.sort_values(ascending=False).head(top_n)

    title_of = dict(zip(books["book_id"], books["title"]))
    author_of = dict(zip(books["book_id"], books["authors"]))
    rows = []
    for book_id, sc in top.items():
        # Build a display-ready row here so app/notebook callers do not each
        # need to repeat title/author joins.
        row = {"book_id": book_id, "title": title_of.get(book_id, "?"),
               "authors": author_of.get(book_id, "?"), "score": round(float(sc), 4)}
        rows.append(row)
    return pd.DataFrame(rows)


# ===== end inlined source =====

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_APP_DIR, ".env"))

# On Streamlit Community Cloud the API key is supplied via the app's Secrets
# (Settings -> Secrets, TOML: GEMINI_API_KEY="..."). Mirror it into the
# environment so the existing os.environ-based re-ranker picks it up; locally the
# .env above still works and this is a harmless no-op.
if not os.environ.get("GEMINI_API_KEY"):
    try:
        _secret_key = st.secrets.get("GEMINI_API_KEY")
        if _secret_key:
            os.environ["GEMINI_API_KEY"] = str(_secret_key)
    except Exception:
        pass


try:
    _require_surprise()
    HAVE_SURPRISE = True
except Exception:
    HAVE_SURPRISE = False

st.set_page_config(
    page_title="BookRec",
    page_icon=":material/auto_stories:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DEFAULT_K = 10
DEFAULT_UBCF_K = 25       # SHIPPED: best memory-light model (UBCF pearson_baseline)
DEFAULT_MIN_RATINGS = 20
BEST_SIM = "pearson_baseline"   # similarity that won the corrected audit

SOURCE_REAL = "Real dataset"

MODEL_LABELS = {
    "ubcf": "User-based CF",
    "ibcf": "Item-based CF",
}


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&display=swap');

        :root {
            --app-bg: #f4ecdd;
            --surface: #fbf6ec;
            --surface-soft: #f0e6d4;
            --text: #2a2118;
            --muted: #8a7c66;
            --line: #e7dcc7;
            --line-strong: #d8cbb1;
            --accent: #a35421;
            --accent-soft: #f3e6d2;
            --shadow: 0 22px 60px rgba(74, 54, 32, 0.16);
            --ink: #2a2118;
            --serif: 'Fraunces', Georgia, 'Times New Roman', serif;
            --paper-edge: #e3d6bd;
            --leather: #7c4a23;
            --leather-dark: #5d3618;
        }

        html, body, .stApp, [class*="css"] {
            color: var(--text);
            font-family: Inter, ui-sans-serif, system-ui, -apple-system,
                BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        .stApp {
            background:
                radial-gradient(1100px 520px at 50% -6%, rgba(214, 158, 86, 0.32), rgba(214, 158, 86, 0) 62%),
                radial-gradient(820px 620px at 50% 24%, rgba(255, 247, 232, 0.55), rgba(255, 247, 232, 0) 70%),
                linear-gradient(180deg, #f6efe0 0%, #efe4d0 52%, #ecdfc8 100%);
            background-attachment: fixed;
        }

        .stApp::before {
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='180' height='180'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.5'/%3E%3C/svg%3E");
            content: "";
            inset: 0;
            mix-blend-mode: multiply;
            opacity: 0.06;
            pointer-events: none;
            position: fixed;
            z-index: 0;
        }

        .stApp::after {
            background: radial-gradient(125% 115% at 50% 30%, rgba(0, 0, 0, 0) 52%, rgba(58, 38, 18, 0.17) 100%);
            content: "";
            inset: 0;
            pointer-events: none;
            position: fixed;
            z-index: 0;
        }

        .block-container {
            position: relative;
            z-index: 1;
        }

        .block-container {
            max-width: 1160px;
            padding-top: 7rem;
            padding-bottom: 5rem;
        }

        #MainMenu,
        footer,
        header[data-testid="stHeader"],
        [data-testid="stDecoration"],
        [data-testid="stToolbar"],
        [data-testid="stHeaderActionElements"],
        .stDeployButton {
            display: none !important;
        }

        section[data-testid="stSidebar"],
        [data-testid="stSidebarNav"] {
            display: none !important;
        }

        .floating-nav {
            align-items: center;
            display: flex;
            justify-content: center;
            left: 0;
            pointer-events: none;
            position: fixed;
            right: 0;
            top: 1rem;
            z-index: 999;
        }

        .floating-nav-inner {
            align-items: center;
            backdrop-filter: blur(18px);
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid rgba(229, 229, 231, 0.96);
            border-radius: 999px;
            box-shadow: 0 18px 60px rgba(17, 17, 19, 0.10);
            display: flex;
            gap: 0.32rem;
            max-width: min(1080px, calc(100vw - 2rem));
            min-height: 4rem;
            padding: 0.46rem;
            pointer-events: auto;
            width: fit-content;
        }

        .top-brand {
            align-items: center;
            display: inline-flex;
            gap: 0.65rem;
            padding: 0.42rem 0.7rem 0.42rem 0.5rem;
            text-decoration: none !important;
        }

        .brand-mark {
            align-items: center;
            background: #111113;
            border-radius: 999px;
            color: #fff;
            display: inline-flex;
            font-size: 0.82rem;
            font-weight: 800;
            height: 2.25rem;
            justify-content: center;
            width: 2.25rem;
        }

        .brand-wordmark {
            color: var(--text);
            display: block;
            font-size: 0.95rem;
            font-weight: 790;
            line-height: 1;
        }

        .brand-caption {
            color: var(--muted);
            display: block;
            font-size: 0.68rem;
            line-height: 1.1;
            margin-top: 0.16rem;
            white-space: nowrap;
        }

        .top-links {
            align-items: center;
            display: flex;
            gap: 0.08rem;
            padding: 0 0.28rem;
        }

        .nav-pill {
            align-items: center;
            border-radius: 999px;
            color: #292a2e !important;
            display: inline-flex;
            font-size: 0.84rem;
            font-weight: 680;
            min-height: 2.55rem;
            padding: 0 0.9rem;
            text-decoration: none !important;
            white-space: nowrap;
        }

        .nav-pill:hover {
            background: #f1f0ef;
        }

        .nav-status {
            align-items: center;
            border-left: 1px solid var(--line);
            color: var(--muted);
            display: inline-flex;
            font-size: 0.78rem;
            font-weight: 680;
            gap: 0.42rem;
            min-height: 2.55rem;
            padding: 0 0.82rem;
            white-space: nowrap;
        }

        .status-dot {
            background: var(--accent);
            border-radius: 999px;
            box-shadow: 0 0 0 4px rgba(15, 118, 110, 0.12);
            height: 0.48rem;
            width: 0.48rem;
        }

        .nav-cta {
            align-items: center;
            background: #111113;
            border-radius: 999px;
            color: #fff !important;
            display: inline-flex;
            font-size: 0.84rem;
            font-weight: 760;
            min-height: 2.55rem;
            padding: 0 1rem;
            text-decoration: none !important;
            white-space: nowrap;
        }

        .page-kicker {
            color: var(--accent);
            font-size: 0.78rem;
            font-weight: 780;
            margin-bottom: 0.72rem;
            text-transform: uppercase;
        }

        .page-title {
            font-size: 3.45rem;
            font-weight: 790;
            letter-spacing: 0;
            line-height: 0.98;
            margin: 0;
            max-width: 840px;
        }

        .page-copy {
            color: var(--muted);
            font-size: 1.02rem;
            line-height: 1.65;
            margin: 1rem 0 1.8rem;
            max-width: 720px;
        }

        .section {
            border-top: 1px solid var(--line);
            margin-top: 2rem;
            padding-top: 2rem;
        }

        .section-title {
            font-size: 1.34rem;
            font-weight: 780;
            letter-spacing: 0;
            margin: 0 0 0.35rem;
        }

        .section-copy {
            color: var(--muted);
            font-size: 0.94rem;
            line-height: 1.55;
            margin-bottom: 1rem;
        }

        .metric-card {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 8px;
            min-height: 112px;
            padding: 1rem;
        }

        .metric-label {
            color: var(--muted);
            font-size: 0.76rem;
            font-weight: 760;
            text-transform: uppercase;
        }

        .metric-value {
            color: var(--text);
            font-size: 1.7rem;
            font-weight: 790;
            line-height: 1.15;
            margin-top: 0.45rem;
        }

        .metric-note {
            color: var(--muted);
            font-size: 0.82rem;
            margin-top: 0.35rem;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: #fff;
            border-color: var(--line) !important;
            border-radius: 8px !important;
            box-shadow: 0 8px 30px rgba(17, 17, 19, 0.04);
        }

        div[data-baseweb="select"] > div,
        input {
            border-color: var(--line-strong) !important;
            border-radius: 8px !important;
        }

        .stSlider label,
        .stSelectbox label,
        .stTextInput label {
            color: #232326;
            font-size: 0.84rem;
            font-weight: 650;
        }

        .stButton > button,
        [data-testid="stBaseButton-primary"],
        [data-testid="stBaseButton-secondary"] {
            border-radius: 8px !important;
            font-weight: 740 !important;
            min-height: 2.75rem;
        }

        [data-testid="stBaseButton-primary"] {
            background: #111113 !important;
            border-color: #111113 !important;
            color: #fff !important;
        }

        [data-testid="stBaseButton-secondary"] {
            background: #fff !important;
            border-color: var(--line-strong) !important;
            color: #111113 !important;
        }

        .pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin: 0.75rem 0 0.2rem;
        }

        .pill {
            align-items: center;
            background: var(--surface-soft);
            border: 1px solid var(--line);
            border-radius: 999px;
            color: #303034;
            display: inline-flex;
            font-size: 0.78rem;
            font-weight: 680;
            padding: 0.28rem 0.62rem;
        }

        .rec-list {
            display: grid;
            gap: 0.78rem;
            margin-top: 0.7rem;
        }

        .rec-card {
            align-items: center;
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 8px;
            display: grid;
            gap: 0.8rem;
            grid-template-columns: 2.35rem 2.5rem minmax(0, 1fr) auto;
            padding: 1rem;
            transition: border-color 140ms ease, box-shadow 140ms ease,
                transform 140ms ease;
        }

        .rec-card:hover {
            border-color: #cfcfd5;
            box-shadow: 0 12px 30px rgba(17, 17, 19, 0.08);
            transform: translateY(-1px);
        }

        .rank {
            align-items: center;
            background: #111113;
            border-radius: 8px;
            color: #fff;
            display: inline-flex;
            font-size: 0.86rem;
            font-weight: 790;
            height: 2.35rem;
            justify-content: center;
            width: 2.35rem;
        }

        .rec-cover {
            background: var(--surface-soft);
            border: 1px solid var(--line);
            border-radius: 4px;
            box-shadow: 0 2px 8px rgba(74, 54, 32, 0.18);
            display: block;
            height: 3.6rem;
            object-fit: cover;
            transition: box-shadow 160ms ease, transform 160ms ease;
            width: 2.5rem;
        }

        .rec-cover--empty {
            align-items: center;
            color: var(--muted);
            display: flex;
            font-size: 1.05rem;
            justify-content: center;
        }

        .rec-cover-link {
            display: block;
            line-height: 0;
        }

        .rec-cover-link:hover .rec-cover {
            box-shadow: 0 7px 18px rgba(74, 54, 32, 0.30);
            transform: translateY(-1px);
        }

        .rec-title {
            color: var(--text);
            font-size: 1rem;
            font-weight: 750;
            line-height: 1.28;
            overflow-wrap: anywhere;
        }

        .rec-title-link {
            color: inherit;
            text-decoration: none;
        }

        .rec-title-link:hover {
            text-decoration: underline;
            text-decoration-color: var(--accent);
            text-underline-offset: 2px;
        }

        .rec-author {
            color: var(--muted);
            font-size: 0.86rem;
            margin-top: 0.24rem;
            overflow-wrap: anywhere;
        }

        .rec-meta {
            color: var(--muted);
            display: flex;
            flex-wrap: wrap;
            font-size: 0.78rem;
            gap: 0.45rem;
            margin-top: 0.62rem;
        }

        .rec-score {
            align-self: start;
            background: var(--accent-soft);
            border: 1px solid #d6f2eb;
            border-radius: 8px;
            color: var(--accent);
            font-size: 0.82rem;
            font-weight: 780;
            padding: 0.45rem 0.62rem;
            white-space: nowrap;
        }

        .empty-state {
            align-items: center;
            background: linear-gradient(180deg, #fff 0%, #fafafa 100%);
            border: 1px dashed var(--line-strong);
            border-radius: 8px;
            color: var(--muted);
            display: flex;
            justify-content: center;
            min-height: 260px;
            padding: 1.5rem;
            text-align: center;
        }

        .empty-title {
            color: var(--text);
            font-size: 1.05rem;
            font-weight: 760;
            margin-bottom: 0.25rem;
        }

        .chat-hero {
            align-items: center;
            display: flex;
            flex-direction: column;
            justify-content: center;
            min-height: clamp(360px, 44vh, 520px);
            padding: clamp(4rem, 9vh, 6.5rem) 0 2.5rem;
            text-align: center;
        }

        .chat-title {
            font-size: clamp(2rem, 3.5vw, 3.1rem);
            font-weight: 500;
            letter-spacing: 0;
            line-height: 1.08;
            margin-bottom: 1.1rem;
        }

        .chat-subtitle {
            color: var(--muted);
            font-size: 1.05rem;
            line-height: 1.7;
            max-width: 820px;
        }

        .chat-lock {
            background: #fff7ed;
            border: 1px solid #fed7aa;
            border-radius: 8px;
            color: #9a3412;
            font-size: 0.94rem;
            margin: 0 auto 2.1rem;
            max-width: 860px;
            padding: 0.95rem 1.2rem;
        }

        .chat-toolbar {
            color: var(--muted);
            font-size: 0.84rem;
            line-height: 1.45;
            margin: 1.4rem auto 0;
            max-width: 720px;
            text-align: center;
        }

        .st-key-chat_stage {
            margin-left: 50%;
            padding: 0 0 3.5rem;
            transform: translateX(-50%);
            width: min(1320px, calc(100vw - 6rem));
        }

        .st-key-chat_composer {
            background: #ffffff;
            border: 1px solid var(--line-strong);
            border-radius: 26px;
            box-shadow: 0 18px 60px rgba(17, 17, 19, 0.10);
            margin: 0 auto;
            max-width: 768px;
            padding: 0.55rem 0.7rem 0.6rem;
            transition: border-color 140ms ease, box-shadow 140ms ease;
        }

        .st-key-chat_composer:focus-within {
            border-color: #c4c4cb;
            box-shadow: 0 22px 72px rgba(17, 17, 19, 0.14);
        }

        .st-key-chat_composer div[data-testid="stVerticalBlock"] {
            gap: 0.15rem;
        }

        .st-key-chat_composer div[data-testid="stHorizontalBlock"] {
            align-items: center;
        }

        .st-key-chat_composer div[data-testid="stTextArea"] textarea {
            background: transparent !important;
            border: 0 !important;
            box-shadow: none !important;
            font-size: 1.04rem;
            line-height: 1.5;
            min-height: 3.4rem !important;
            padding: 0.85rem 1rem 0.45rem !important;
            resize: none;
        }

        .st-key-chat_composer div[data-baseweb="select"] > div {
            background: #f4f4f5 !important;
            border-color: #ececef !important;
            border-radius: 999px !important;
            min-height: 2.9rem;
        }

        .st-key-chat_composer .stButton > button {
            border-radius: 999px !important;
            min-height: 2.9rem;
        }

        .st-key-chat_suggestions {
            margin: 1.1rem auto 0;
            max-width: 768px;
        }

        .st-key-chat_suggestions .stButton > button {
            background: #fff !important;
            border-color: var(--line-strong) !important;
            border-radius: 999px !important;
            box-shadow: 0 8px 24px rgba(17, 17, 19, 0.04);
            min-height: 3.45rem;
            padding-left: 1.1rem !important;
            padding-right: 1.1rem !important;
            white-space: normal;
        }

        /* Clarify answer chips — rendered directly under the question, above the
           text box, with an accent so they read as "tap to answer". */
        .st-key-chat_clarify {
            margin: 0.4rem auto 0.5rem;
            max-width: 768px;
        }

        .clarify-answer-label {
            color: var(--accent);
            font-size: 0.82rem;
            font-weight: 760;
            letter-spacing: 0.01em;
            margin: 0 auto 0.6rem;
            max-width: 768px;
            text-align: center;
        }

        .st-key-chat_clarify .stButton > button {
            background: var(--surface) !important;
            border: 1.5px solid var(--accent) !important;
            border-radius: 999px !important;
            box-shadow: 0 8px 24px rgba(163, 84, 33, 0.10);
            color: var(--ink) !important;
            font-weight: 720 !important;
            min-height: 3.1rem;
            padding-left: 1.05rem !important;
            padding-right: 1.05rem !important;
            white-space: normal;
        }

        .st-key-chat_clarify .stButton > button:hover:not(:disabled) {
            background: var(--accent-soft) !important;
            border-color: var(--accent) !important;
            box-shadow: 0 12px 30px rgba(163, 84, 33, 0.16);
            transform: translateY(-1px);
        }

        .chat-thread {
            display: flex;
            flex-direction: column;
            gap: 2.4rem;
            margin: 0.5rem auto 2.6rem;
            max-width: 720px;
            width: 100%;
        }

        .msg {
            display: flex;
            width: 100%;
        }

        .msg-user {
            justify-content: flex-end;
        }

        .bubble-user {
            background: var(--surface-soft);
            border: 1px solid var(--line);
            border-radius: 1.35rem 1.35rem 0.45rem 1.35rem;
            color: #2f261b;
            font-size: 0.97rem;
            line-height: 1.6;
            max-width: 78%;
            overflow-wrap: anywhere;
            padding: 0.7rem 1.1rem;
            white-space: pre-wrap;
        }

        .msg-assistant {
            align-items: flex-start;
            gap: 0.95rem;
            justify-content: flex-start;
        }

        .assistant-avatar {
            align-items: center;
            background: var(--ink);
            border-radius: 50%;
            color: #fbf6ec;
            display: inline-flex;
            flex: 0 0 auto;
            font-family: var(--serif);
            font-size: 0.82rem;
            font-weight: 600;
            height: 1.95rem;
            justify-content: center;
            margin-top: 0.1rem;
            width: 1.95rem;
        }

        .assistant-body {
            flex: 1 1 auto;
            min-width: 0;
        }

        .assistant-name {
            color: var(--text);
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.01em;
        }

        .assistant-lead {
            color: #4a4031;
            font-size: 1.04rem;
            line-height: 1.62;
            margin-top: 0.35rem;
        }

        .assistant-meta {
            color: var(--muted);
            font-size: 0.74rem;
            margin-top: 0.9rem;
        }

        .thinking {
            align-items: center;
            display: inline-flex;
            gap: 0.34rem;
            margin-top: 0.6rem;
        }

        .thinking span {
            animation: thinking-bounce 1.2s infinite ease-in-out;
            background: #b9b6ad;
            border-radius: 50%;
            height: 0.5rem;
            width: 0.5rem;
        }

        .thinking span:nth-child(2) {
            animation-delay: 0.18s;
        }

        .thinking span:nth-child(3) {
            animation-delay: 0.36s;
        }

        @keyframes thinking-bounce {
            0%, 80%, 100% {
                opacity: 0.35;
                transform: translateY(0);
            }
            40% {
                opacity: 1;
                transform: translateY(-0.28rem);
            }
        }

        .pick-list {
            display: grid;
            gap: 0.85rem;
            margin-top: 1.25rem;
        }

        .pick-card {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 14px;
            display: grid;
            gap: 0.05rem 0.85rem;
            grid-template-columns: 2.4rem 2.8rem minmax(0, 1fr);
            padding: 1.05rem 1.15rem;
            transition: border-color 160ms ease, box-shadow 160ms ease,
                transform 160ms ease;
        }

        .pick-card:hover {
            border-color: var(--line-strong);
            box-shadow: 0 14px 32px rgba(74, 54, 32, 0.12);
            transform: translateY(-1px);
        }

        .pick-rank {
            align-items: center;
            background: var(--ink);
            border-radius: 50%;
            color: #fbf6ec;
            display: inline-flex;
            font-family: var(--serif);
            font-size: 0.92rem;
            font-weight: 600;
            grid-column: 1;
            grid-row: 1 / span 3;
            height: 2.4rem;
            justify-content: center;
            margin-top: 0.05rem;
            width: 2.4rem;
        }

        .pick-cover {
            align-self: start;
            display: block;
            grid-column: 2;
            grid-row: 1 / span 3;
            margin-top: 0.12rem;
        }

        .pick-cover-img {
            border-radius: 5px;
            box-shadow: 0 3px 10px rgba(74, 54, 32, 0.22);
            display: block;
            height: 4.2rem;
            object-fit: cover;
            transition: box-shadow 160ms ease, transform 160ms ease;
            width: 2.8rem;
        }

        .pick-cover:hover .pick-cover-img {
            box-shadow: 0 7px 18px rgba(74, 54, 32, 0.30);
            transform: translateY(-1px);
        }

        .pick-cover--empty {
            align-items: center;
            background: var(--surface-soft);
            border: 1px solid var(--line);
            box-shadow: none;
            color: var(--muted);
            display: flex;
            font-size: 1.1rem;
            justify-content: center;
        }

        .pick-title {
            color: var(--text);
            font-size: 1.04rem;
            font-weight: 600;
            grid-column: 3;
            line-height: 1.3;
            overflow-wrap: anywhere;
        }

        .pick-title-link {
            color: inherit;
            text-decoration: none;
        }

        .pick-title-link:hover {
            text-decoration: underline;
            text-decoration-color: var(--accent);
            text-underline-offset: 2px;
        }

        .pick-author {
            color: var(--muted);
            font-size: 0.85rem;
            font-weight: 500;
        }

        .pick-desc {
            color: #4a4031;
            font-size: 0.92rem;
            grid-column: 3;
            line-height: 1.58;
            margin-top: 0.5rem;
        }

        .pick-why {
            border-left: 2px solid var(--accent);
            color: #5f5340;
            font-size: 0.9rem;
            grid-column: 3;
            line-height: 1.55;
            margin-top: 0.7rem;
            padding-left: 0.8rem;
        }

        .pick-why-label {
            color: var(--accent);
            display: block;
            font-size: 0.68rem;
            font-weight: 780;
            letter-spacing: 0.06em;
            margin-bottom: 0.2rem;
            text-transform: uppercase;
        }

        .pick-empty {
            color: #5f5340;
            font-size: 0.92rem;
            line-height: 1.55;
            margin-top: 0.6rem;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid var(--line);
            border-radius: 8px;
            overflow: hidden;
        }

        #filtering,
        #candidates,
        #quality,
        #personalize {
            scroll-margin-top: 6rem;
        }

        @media (max-width: 700px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
                padding-top: 5.8rem;
            }

            .floating-nav {
                justify-content: flex-start;
                overflow-x: auto;
                padding: 0 0.7rem;
            }

            .floating-nav-inner {
                max-width: none;
                min-width: max-content;
            }

            .top-brand {
                padding-right: 0.15rem;
            }

            .brand-wordmark,
            .brand-caption,
            .nav-status,
            .nav-cta {
                display: none;
            }

            .nav-pill {
                min-height: 2.4rem;
                padding: 0 0.68rem;
            }

            .page-title {
                font-size: 2.35rem;
            }

            .chat-hero {
                min-height: 300px;
                padding: 3rem 0 1.75rem;
            }

            .chat-subtitle {
                font-size: 0.98rem;
            }

            .st-key-chat_stage {
                margin-left: 0;
                padding-bottom: 2.5rem;
                transform: none;
                width: auto;
            }

            .st-key-chat_composer {
                border-radius: 24px;
                padding: 0.5rem;
            }

            .st-key-chat_composer div[data-testid="stHorizontalBlock"] > div:first-child {
                display: none;
            }

            .st-key-chat_composer div[data-testid="stTextArea"] textarea {
                min-height: 6rem !important;
            }

            .st-key-chat_suggestions .stButton > button {
                min-height: 3.2rem;
            }

            .rec-card {
                grid-template-columns: 2.1rem 2.3rem minmax(0, 1fr);
            }

            .rec-score {
                grid-column: 3;
                justify-self: start;
            }
        }
        /* ============ Warm library + cinematic layer ============ */
        .page-title,
        .chat-title,
        .section-title,
        .brand-wordmark,
        .metric-value,
        .empty-title,
        .rec-title,
        .pick-title {
            font-family: var(--serif);
        }

        .page-title,
        .chat-title {
            font-weight: 600;
            letter-spacing: -0.012em;
        }

        .section-title {
            font-weight: 600;
        }

        .floating-nav-inner {
            background: rgba(251, 246, 236, 0.86);
            border-color: rgba(216, 203, 177, 0.92);
            box-shadow: 0 18px 50px rgba(74, 54, 32, 0.18);
        }

        .brand-mark,
        .rank,
        .assistant-avatar {
            background: var(--ink);
            color: #fbf6ec;
        }

        .nav-pill {
            color: #3c3326 !important;
        }

        .nav-pill:hover {
            background: #ece1cb;
        }

        .nav-cta {
            background: var(--ink);
            color: #fbf6ec !important;
        }

        .status-dot {
            background: var(--accent);
            box-shadow: 0 0 0 4px rgba(163, 84, 33, 0.16);
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--surface);
            box-shadow: 0 16px 42px rgba(74, 54, 32, 0.12);
        }

        [data-testid="stBaseButton-primary"] {
            background: var(--ink) !important;
            border-color: var(--ink) !important;
            color: #fbf6ec !important;
        }

        [data-testid="stBaseButton-secondary"] {
            background: var(--surface) !important;
            border-color: var(--line-strong) !important;
            color: var(--ink) !important;
        }

        .pill {
            background: var(--surface-soft);
            color: #4a4031;
        }

        .metric-card,
        .rec-card {
            background: var(--surface);
        }

        .rec-card:hover {
            border-color: var(--line-strong);
            box-shadow: 0 18px 42px rgba(74, 54, 32, 0.16);
        }

        .rec-score {
            background: var(--accent-soft);
            border-color: #e7d2b4;
            color: var(--accent);
        }

        .empty-state {
            background: linear-gradient(180deg, var(--surface) 0%, #f3ead9 100%);
        }

        .st-key-chat_composer {
            background: var(--surface);
        }

        .st-key-chat_composer:focus-within {
            border-color: var(--accent);
            box-shadow: 0 22px 64px rgba(163, 84, 33, 0.20);
        }

        .st-key-chat_composer div[data-baseweb="select"] > div {
            background: var(--surface-soft) !important;
            border-color: var(--line) !important;
        }

        .st-key-chat_suggestions .stButton > button {
            background: var(--surface) !important;
            box-shadow: 0 10px 26px rgba(74, 54, 32, 0.10);
        }

        .bubble-user {
            background: #ece0ca;
            color: #2f261b;
        }

        .thinking span {
            background: #b9a888;
        }

        .stSlider label,
        .stSelectbox label,
        .stTextInput label {
            color: #3c3326;
        }

        .chat-lock {
            background: #f6e7d0;
            border-color: #e6c79a;
            color: #8a3f12;
        }

        /* --- Cinematic hero --- */
        .cine-hero {
            align-items: center;
            display: flex;
            flex-direction: column;
            padding: 0.5rem 0 1.25rem;
            position: relative;
            text-align: center;
        }

        .cine-hero .page-kicker {
            letter-spacing: 0.14em;
            margin-top: 0.5rem;
        }

        .cine-hero .page-title,
        .cine-hero .page-copy {
            margin-left: auto;
            margin-right: auto;
        }

        /* --- Flipping book --- */
        .book {
            --w: clamp(74px, 9vw, 118px);
            --h: clamp(106px, 13vw, 170px);
            --dur: 8s;
            height: var(--h);
            perspective: 1500px;
            position: relative;
            width: calc(var(--w) * 2);
        }

        .book-hero {
            animation: book-float 6s ease-in-out infinite;
            filter: drop-shadow(0 26px 28px rgba(60, 38, 16, 0.28));
            margin: 0 auto 1.5rem;
        }

        @keyframes book-float {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-9px); }
        }

        .book-3d {
            height: 100%;
            position: relative;
            transform: rotateX(13deg);
            transform-style: preserve-3d;
            width: 100%;
        }

        .book-cover {
            background: linear-gradient(145deg, var(--leather) 0%, var(--leather-dark) 100%);
            border-radius: 5px 9px 9px 5px;
            bottom: -7px;
            box-shadow: inset 0 0 0 1px rgba(0, 0, 0, 0.2);
            left: -7px;
            position: absolute;
            right: -7px;
            top: -7px;
        }

        .page-static {
            background: linear-gradient(180deg, #fdf9ef 0%, #f1e6cf 100%);
            border: 1px solid var(--paper-edge);
            height: 100%;
            position: absolute;
            top: 0;
            width: var(--w);
        }

        .page-left { border-radius: 3px 0 0 3px; left: 0; }
        .page-right { border-radius: 0 3px 3px 0; right: 0; }

        .spine {
            background: linear-gradient(90deg, rgba(60, 36, 14, 0.30), rgba(60, 36, 14, 0));
            height: 100%;
            left: calc(var(--w) - 3px);
            position: absolute;
            top: 0;
            width: 7px;
            z-index: 8;
        }

        .leaf {
            animation: leaf-riffle var(--dur) ease-in-out infinite;
            background: linear-gradient(180deg, #fdfaf2 0%, #f3e9d4 100%);
            border: 1px solid var(--paper-edge);
            border-radius: 0 3px 3px 0;
            height: 100%;
            left: var(--w);
            position: absolute;
            top: 0;
            transform-origin: left center;
            width: var(--w);
        }

        .leaf::after {
            background: linear-gradient(90deg, rgba(0, 0, 0, 0) 68%, rgba(70, 44, 18, 0.12) 100%);
            content: "";
            inset: 0;
            position: absolute;
        }

        .leaf-1 { animation-delay: 0s; }
        .leaf-2 { animation-delay: 1s; }
        .leaf-3 { animation-delay: 2s; }
        .leaf-4 { animation-delay: 3s; }

        @keyframes leaf-riffle {
            0% { transform: rotateY(0deg); }
            42% { transform: rotateY(-172deg); }
            52% { transform: rotateY(-172deg); }
            94% { transform: rotateY(0deg); }
            100% { transform: rotateY(0deg); }
        }

        /* --- Ambient side books --- */
        .side-book {
            display: none;
            filter: drop-shadow(0 14px 18px rgba(60, 38, 16, 0.24));
            opacity: 0.6;
            position: fixed;
            top: 50%;
            transform: translateY(-50%);
            z-index: 1;
        }

        .book-side { --w: 46px; --h: 66px; --dur: 9s; }
        .side-left { left: 2rem; }
        .side-right { right: 2rem; }
        .side-right.book-side { --dur: 10.5s; }

        @media (min-width: 1440px) {
            .side-book { display: block; }
        }

        /* --- Image artifact panels (fixed; wide screens only) --- */
        .art-panel {
            display: none;
            height: 100vh;
            opacity: 0.9;
            pointer-events: none;
            position: fixed;
            top: 0;
            width: clamp(132px, calc((100vw - 1180px) / 2 - 6px), 360px);
            z-index: 0;
        }

        .art-panel img {
            -webkit-mask-composite: source-in;
            -webkit-mask-image:
                linear-gradient(to right, transparent 0, #000 20%, #000 80%, transparent 100%),
                linear-gradient(to bottom, transparent 0, #000 11%, #000 89%, transparent 100%);
            display: block;
            height: 100%;
            mask-composite: intersect;
            mask-image:
                linear-gradient(to right, transparent 0, #000 20%, #000 80%, transparent 100%),
                linear-gradient(to bottom, transparent 0, #000 11%, #000 89%, transparent 100%);
            object-fit: cover;
            width: 100%;
        }

        .art-left { left: 0; }
        .art-right { right: 0; }

        .art-left img {
            animation: spiral-sway 12s ease-in-out infinite;
            transform-origin: 50% 42%;
            will-change: transform;
        }

        @keyframes spiral-sway {
            0%, 100% { transform: rotate(-2deg) translateY(0) scale(1); }
            50% { transform: rotate(2deg) translateY(-12px) scale(1.03); }
        }

        .art-right img {
            animation: node-pulse 5.5s ease-in-out infinite;
            will-change: transform, filter, opacity;
        }

        @keyframes node-pulse {
            0%, 100% {
                filter: brightness(0.97) drop-shadow(0 0 0 rgba(206, 128, 66, 0));
                opacity: 0.86;
                transform: translateY(0);
            }
            50% {
                filter: brightness(1.13) drop-shadow(0 0 11px rgba(206, 128, 66, 0.5));
                opacity: 1;
                transform: translateY(-7px);
            }
        }

        @media (min-width: 1400px) {
            .art-panel { display: block; }
        }

        /* --- Image book centerpiece with flipping pages --- */
        .center-stage {
            animation: book-float 6s ease-in-out infinite;
            filter: drop-shadow(0 20px 24px rgba(60, 38, 16, 0.20));
            margin: 0 auto 1.4rem;
            position: relative;
            width: clamp(232px, 30vw, 360px);
        }

        .center-book-img {
            display: block;
            height: auto;
            width: 100%;
        }

        .flip-layer {
            bottom: 17%;
            left: 8%;
            perspective: 1700px;
            position: absolute;
            right: 8%;
            top: 16%;
            transform-style: preserve-3d;
        }

        .img-leaf {
            animation: page-flip 7s ease-in-out infinite;
            background: linear-gradient(95deg, #e6dcc6 0%, #efe8d6 18%, #ece4d2 100%);
            border: 1px solid rgba(120, 90, 50, 0.16);
            border-left-color: rgba(120, 90, 50, 0.30);
            bottom: 0;
            box-shadow: -2px 0 6px rgba(60, 40, 18, 0.12);
            left: 50%;
            position: absolute;
            top: 0;
            transform-origin: left center;
            width: 49%;
        }

        .img-leaf::before {
            border: 1px solid rgba(120, 90, 50, 0.14);
            content: "";
            inset: 7% 9%;
            position: absolute;
        }

        .leaf-a { animation-delay: 0s; }
        .leaf-b { animation-delay: 1.6s; }
        .leaf-c { animation-delay: 3.2s; }

        @keyframes page-flip {
            0% { transform: rotateY(0deg); }
            46% { transform: rotateY(-167deg); }
            54% { transform: rotateY(-167deg); }
            100% { transform: rotateY(0deg); }
        }

        /* ============ Sequential DAG: intent chips + reasoning trace ============ */
        .intent-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
            margin: 0.65rem 0 0.2rem;
        }

        .intent-chip {
            align-items: center;
            background: var(--accent-soft);
            border: 1px solid #e7d2b4;
            border-radius: 999px;
            color: var(--accent);
            display: inline-flex;
            font-size: 0.72rem;
            font-weight: 760;
            letter-spacing: 0.03em;
            padding: 0.24rem 0.6rem;
            text-transform: uppercase;
        }

        .intent-chip--avoid {
            background: var(--surface-soft);
            border-color: var(--line-strong);
            color: var(--muted);
            text-decoration: line-through;
            text-decoration-thickness: 1px;
        }

        .reasoning-trace {
            background: linear-gradient(180deg, #fefcf6 0%, #fbf6ec 100%);
            border: 1px solid var(--line);
            border-radius: 14px;
            margin-top: 1rem;
            overflow: hidden;
            transition: border-color 160ms ease, box-shadow 160ms ease;
        }

        .reasoning-trace[open] {
            border-color: var(--line-strong);
            box-shadow: 0 16px 38px rgba(74, 54, 32, 0.12);
        }

        .reasoning-trace > summary {
            align-items: center;
            color: var(--ink);
            cursor: pointer;
            display: flex;
            font-size: 0.86rem;
            font-weight: 680;
            gap: 0.5rem;
            letter-spacing: 0.01em;
            list-style: none;
            padding: 0.85rem 1.05rem;
            user-select: none;
        }

        .reasoning-trace > summary:hover {
            background: rgba(163, 84, 33, 0.05);
        }

        .reasoning-trace > summary::-webkit-details-marker {
            display: none;
        }

        .reasoning-trace > summary::before {
            color: var(--accent);
            content: "›";
            display: inline-block;
            font-size: 1.1rem;
            font-weight: 800;
            transition: transform 160ms ease;
        }

        .reasoning-trace[open] > summary::before {
            transform: rotate(90deg);
        }

        .trace-intro {
            border-top: 1px solid var(--line);
            color: var(--muted);
            font-size: 0.82rem;
            line-height: 1.5;
            padding: 0.7rem 1.05rem 0.2rem;
        }

        .trace-stages {
            counter-reset: trace-step;
            padding: 0.4rem 1.05rem 0.9rem;
            position: relative;
        }

        /* vertical connector running through the numbered steps */
        .trace-stages::before {
            background: var(--line-strong);
            bottom: 1.7rem;
            content: "";
            left: calc(1.05rem + 0.86rem);
            position: absolute;
            top: 1.4rem;
            width: 2px;
            z-index: 0;
        }

        .trace-stage {
            padding: 0.7rem 0 0.7rem 2.55rem;
            position: relative;
            z-index: 1;
        }

        .trace-stage-head {
            align-items: center;
            display: flex;
            gap: 0.55rem;
        }

        .trace-step-no {
            align-items: center;
            background: var(--surface);
            border: 2px solid var(--line-strong);
            border-radius: 50%;
            color: var(--accent);
            display: inline-flex;
            font-size: 0.78rem;
            font-weight: 780;
            height: 1.72rem;
            justify-content: center;
            left: 0;
            position: absolute;
            top: 0.62rem;
            width: 1.72rem;
        }

        .trace-stage-name {
            color: var(--text);
            flex: 1 1 auto;
            font-family: var(--serif);
            font-size: 1.02rem;
            font-weight: 600;
        }

        .trace-note {
            color: #6b5d48;
            font-size: 0.84rem;
            line-height: 1.5;
            margin-top: 0.35rem;
        }

        .trace-body {
            color: #5f5340;
            font-size: 0.84rem;
            line-height: 1.55;
            margin-top: 0.6rem;
        }

        .trace-line {
            overflow-wrap: anywhere;
            padding: 0.16rem 0;
        }

        .trace-head {
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 700;
            margin-bottom: 0.15rem;
            text-transform: uppercase;
        }

        .trace-chips {
            display: flex;
            flex-wrap: wrap;
            gap: 0.34rem;
            margin-top: 0.1rem;
        }

        .trace-chip {
            background: var(--surface-soft);
            border: 1px solid var(--line);
            border-radius: 999px;
            color: #4a4031;
            font-size: 0.74rem;
            font-weight: 620;
            padding: 0.18rem 0.55rem;
        }

        .trace-chip--avoid {
            color: var(--muted);
            text-decoration: line-through;
            text-decoration-thickness: 1px;
        }

        .trace-strength {
            background: var(--accent-soft);
            border-radius: 6px;
            color: var(--accent);
            font-size: 0.72rem;
            font-weight: 740;
            padding: 0.1rem 0.4rem;
            white-space: nowrap;
        }

        .trace-rank {
            background: var(--ink);
            border-radius: 5px;
            color: #fbf6ec;
            display: inline-block;
            font-size: 0.72rem;
            font-weight: 760;
            min-width: 1.1rem;
            padding: 0.04rem 0.3rem;
            text-align: center;
        }

        .trace-book {
            color: var(--text);
            font-weight: 680;
        }

        .trace-flag {
            color: var(--muted);
            font-size: 0.76rem;
            font-style: italic;
        }

        .clarify-hint {
            color: var(--muted);
            font-size: 0.84rem;
            line-height: 1.5;
            margin-top: 0.7rem;
        }

        /* --- pending stage-progress strip --- */
        .stage-progress {
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
            margin: 0.7rem 0 0.2rem;
        }

        .stage-pill {
            align-items: center;
            background: var(--surface-soft);
            border: 1px solid var(--line);
            border-radius: 999px;
            color: #4a4031;
            display: inline-flex;
            font-size: 0.74rem;
            font-weight: 720;
            gap: 0.4rem;
            padding: 0.26rem 0.62rem;
        }

        .stage-pill-dot {
            animation: thinking-bounce 1.2s infinite ease-in-out;
            background: var(--accent);
            border-radius: 50%;
            height: 0.42rem;
            width: 0.42rem;
        }

        .stage-pill:nth-child(2) .stage-pill-dot {
            animation-delay: 0.18s;
        }

        .stage-pill:nth-child(3) .stage-pill-dot {
            animation-delay: 0.36s;
        }

        @media (prefers-reduced-motion: reduce) {
            .book-hero,
            .leaf,
            .center-stage,
            .img-leaf,
            .art-left img,
            .art-right img,
            .stage-pill-dot {
                animation: none !important;
            }

            .leaf-2 { transform: rotateY(-28deg); }
            .leaf-3 { transform: rotateY(-150deg); }
            .img-leaf { display: none; }
        }

        /* ===================== Premium chat polish ===================== */
        /* Slightly airier thread + crisper reading rhythm. */
        .chat-thread {
            gap: 2.1rem;
        }

        .chat-title {
            letter-spacing: -0.015em;
        }

        .chat-subtitle {
            color: #6b5d48;
        }

        /* Assistant identity — a warm gradient medallion with a soft ring. */
        .assistant-avatar {
            background: linear-gradient(150deg, #6b4423 0%, var(--ink) 100%);
            box-shadow:
                0 4px 12px rgba(42, 33, 24, 0.26),
                0 0 0 3px rgba(163, 84, 33, 0.10);
            font-family: var(--serif);
            letter-spacing: 0.01em;
        }

        .assistant-name {
            letter-spacing: 0.01em;
        }

        /* User turn — a soft, layered paper bubble. */
        .bubble-user {
            background: linear-gradient(180deg, #f1e6d1 0%, #ece0ca 100%);
            border: 1px solid var(--paper-edge);
            border-radius: 1.5rem 1.5rem 0.5rem 1.5rem;
            box-shadow: 0 2px 8px rgba(74, 54, 32, 0.08);
        }

        /* Pick cards — layered depth, soft inner highlight, lifted hover. */
        .pick-card {
            background: linear-gradient(180deg, #fdf9f0 0%, var(--surface) 100%);
            border-color: var(--line);
            border-radius: 16px;
            box-shadow:
                0 1px 0 rgba(255, 255, 255, 0.6) inset,
                0 6px 18px rgba(74, 54, 32, 0.07);
            padding: 1.15rem 1.25rem;
            transition: border-color 180ms ease, box-shadow 180ms ease,
                transform 180ms ease;
        }

        .pick-card:hover {
            border-color: var(--line-strong);
            box-shadow:
                0 1px 0 rgba(255, 255, 255, 0.6) inset,
                0 18px 40px rgba(74, 54, 32, 0.16);
            transform: translateY(-2px);
        }

        .pick-rank {
            background: linear-gradient(150deg, #6b4423 0%, var(--ink) 100%);
            box-shadow:
                0 4px 12px rgba(42, 33, 24, 0.24),
                0 0 0 3px rgba(163, 84, 33, 0.08);
        }

        /* The "why it ranks here" line — a tinted, inset rationale chip. */
        .pick-why {
            background: linear-gradient(180deg, #fbeedd 0%, var(--accent-soft) 100%);
            border-left: 2px solid var(--accent);
            border-radius: 0 9px 9px 0;
            margin-top: 0.8rem;
            padding: 0.55rem 0.75rem;
        }

        .pick-why-label {
            letter-spacing: 0.07em;
        }

        /* Reasoning disclosure — quieter, rounder, premium. */
        .reasoning-trace {
            border-radius: 16px;
        }

        .reasoning-trace > summary {
            font-weight: 640;
            letter-spacing: 0.01em;
        }

        /* Composer — gradient surface, layered shadow, accent focus ring. */
        .st-key-chat_composer {
            background: linear-gradient(180deg, #fffdf8 0%, var(--surface) 100%);
            border-color: var(--line);
            border-radius: 28px;
            box-shadow:
                0 1px 0 rgba(255, 255, 255, 0.7) inset,
                0 2px 6px rgba(74, 54, 32, 0.06),
                0 24px 60px rgba(74, 54, 32, 0.12);
            padding: 0.7rem 0.85rem 0.75rem;
        }

        .st-key-chat_composer:focus-within {
            border-color: var(--accent);
            box-shadow:
                0 0 0 3px rgba(163, 84, 33, 0.12),
                0 26px 72px rgba(163, 84, 33, 0.18);
        }

        .st-key-chat_composer div[data-testid="stTextArea"] textarea::placeholder {
            color: #ab9d85;
        }

        .st-key-chat_composer div[data-baseweb="select"] > div {
            background: var(--surface-soft) !important;
            border-color: var(--line) !important;
            box-shadow: none !important;
        }

        /* Send — circular premium primary with a confident lift. */
        .st-key-chat_composer div[data-testid="column"]:last-child .stButton > button {
            background: linear-gradient(180deg, #3a2c1c 0%, var(--ink) 100%) !important;
            border: 0 !important;
            border-radius: 999px !important;
            box-shadow: 0 6px 18px rgba(42, 33, 24, 0.26);
            color: #fbf6ec !important;
            font-weight: 760 !important;
            letter-spacing: 0.02em;
            transition: transform 150ms ease, box-shadow 150ms ease,
                filter 150ms ease;
        }

        .st-key-chat_composer div[data-testid="column"]:last-child
            .stButton > button:hover:not(:disabled) {
            box-shadow: 0 10px 28px rgba(42, 33, 24, 0.34);
            filter: brightness(1.07);
            transform: translateY(-1px);
        }

        .st-key-chat_composer div[data-testid="column"]:last-child
            .stButton > button:disabled {
            box-shadow: none;
            opacity: 0.45;
        }

        /* Suggestion chips — refined, responsive hover. */
        .st-key-chat_suggestions .stButton > button {
            transition: transform 150ms ease, box-shadow 150ms ease,
                border-color 150ms ease;
        }

        .st-key-chat_suggestions .stButton > button:hover {
            border-color: var(--accent) !important;
            box-shadow: 0 12px 30px rgba(74, 54, 32, 0.14) !important;
            transform: translateY(-1px);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def model_label(kind: str) -> str:
    return MODEL_LABELS.get(kind, kind)


# Small display-formatting helpers keep HTML rendering defensive: app data may
# include None/NaN values from CSVs.
def safe_text(value, fallback: str = "") -> str:
    if value is None:
        return fallback
    try:
        if value != value:
            return fallback
    except TypeError:
        pass
    text = str(value).strip()
    return text or fallback


def format_year(value) -> str:
    try:
        year = int(float(value))
    except (TypeError, ValueError):
        return "Year unknown"
    return str(year) if year > 0 else "Year unknown"


def format_number(value) -> str:
    try:
        return f"{int(float(value)):,}"
    except (TypeError, ValueError):
        return "0"


def format_score(value) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "0.000"


def render_recommendation_cards(recs, books) -> None:
    meta_cols = [
        "book_id",
        "original_publication_year",
        "average_rating",
        "ratings_count",
    ]
    # Pull the cover-image link straight from the catalog when present (the
    # assignment dataset ships Goodreads `small_image_url`/`image_url`; books
    # without one fall back to a placeholder tile).
    for col in ("small_image_url", "image_url"):
        if col in books.columns and col not in meta_cols:
            meta_cols.append(col)
    display = recs.merge(books[meta_cols], on="book_id", how="left")

    # Build one HTML block and send it once; this avoids Streamlit inserting
    # extra vertical gaps between recommendation cards.
    cards = ['<div class="rec-list">']
    for rank, row in enumerate(display.itertuples(index=False), start=1):
        title_raw = safe_text(getattr(row, "title", None), "Untitled")
        authors_raw = safe_text(getattr(row, "authors", None), "Unknown author")
        title = escape(title_raw)
        authors = escape(authors_raw)
        year = escape(format_year(getattr(row, "original_publication_year", None)))
        avg = format_score(getattr(row, "average_rating", None))
        rating_count = escape(format_number(getattr(row, "ratings_count", None)))
        score = escape(format_score(getattr(row, "score", None)))
        cover = safe_text(getattr(row, "small_image_url", None)) or safe_text(
            getattr(row, "image_url", None)
        )
        # Same cover + Goodreads link treatment as the chat pick cards.
        link = escape(_goodreads_url(cover, title_raw, authors_raw))
        if cover.startswith("http"):
            cover_inner = (
                f'<img class="rec-cover" src="{escape(cover)}" alt="" '
                f'loading="lazy" referrerpolicy="no-referrer">'
            )
        else:
            cover_inner = (
                '<span class="rec-cover rec-cover--empty" aria-hidden="true">📖</span>'
            )
        cards.append(
            f'<article class="rec-card">'
            f'<div class="rank">{rank}</div>'
            f'<a class="rec-cover-link" href="{link}" target="_blank" '
            f'rel="noopener noreferrer">{cover_inner}</a>'
            f'<div>'
            f'<div class="rec-title">'
            f'<a class="rec-title-link" href="{link}" target="_blank" '
            f'rel="noopener noreferrer">{title}</a></div>'
            f'<div class="rec-author">{authors}</div>'
            f'<div class="rec-meta">'
            f'<span>{year}</span>'
            f'<span>Avg {avg}</span>'
            f'<span>{rating_count} ratings</span>'
            f'</div>'
            f'</div>'
            f'<div class="rec-score">Score {score}</div>'
            f'</article>'
        )
    cards.append("</div>")
    st.markdown("\n".join(cards), unsafe_allow_html=True)


def render_intent_chips(intent) -> str:
    """A row of pills summarizing the DAG's extracted intent (Stage A).

    `intent` is the asdict() of rag_pipeline.Intent. Returns "" when there is
    nothing meaningful to show so old messages stay clean.
    """
    if not intent:
        return ""
    chips = []
    mood = safe_text(intent.get("mood"))
    if mood:
        chips.append(f'<span class="intent-chip">{escape(mood)}</span>')
    pace = safe_text(intent.get("pace"))
    if pace and pace.lower() != "any":
        chips.append(f'<span class="intent-chip">{escape(pace)} pace</span>')
    recency = safe_text(intent.get("recency"))
    if recency and recency.lower() != "any":
        chips.append(f'<span class="intent-chip">{escape(recency)}</span>')
    for genre in (intent.get("genres") or [])[:3]:
        label = safe_text(genre)
        if label:
            chips.append(f'<span class="intent-chip">{escape(label)}</span>')
    for theme in (intent.get("themes") or [])[:2]:
        label = safe_text(theme)
        if label:
            chips.append(f'<span class="intent-chip">{escape(label)}</span>')
    for avoid in (intent.get("avoid") or [])[:3]:
        label = safe_text(avoid)
        if label:
            chips.append(
                f'<span class="intent-chip intent-chip--avoid">no {escape(label)}</span>'
            )
    if not chips:
        return ""
    return '<div class="intent-row">' + "".join(chips) + "</div>"


def _match_strength(value) -> str:
    """Turn a 0..1 relevance score into a plain-language match label."""
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = 0.0
    if score >= 0.66:
        return "Strong match"
    if score >= 0.33:
        return "Fair match"
    return "Light match"


def _chip_line(label: str) -> str:
    return f'<span class="trace-chip">{escape(label)}</span>'


def _trace_output_summary(name: str, output: dict) -> str:
    """Plain-language view of one step's result, written for a general reader."""
    if not output:
        return ""
    if name == "Understand your request":
        # Stage A output is best shown as compact facets rather than raw JSON.
        chips = []
        mood = safe_text(output.get("mood"))
        if mood:
            chips.append(_chip_line(mood))
        for key in ("pace", "recency"):
            val = safe_text(output.get(key))
            if val and val.lower() != "any":
                chips.append(_chip_line(f"{val} {key}" if key == "pace" else val))
        for genre in (output.get("genres") or []):
            label = safe_text(genre)
            if label:
                chips.append(_chip_line(label))
        for theme in (output.get("themes") or [])[:3]:
            label = safe_text(theme)
            if label:
                chips.append(_chip_line(label))
        for avoid in (output.get("avoid") or []):
            label = safe_text(avoid)
            if label:
                chips.append(
                    f'<span class="trace-chip trace-chip--avoid">avoid {escape(label)}</span>'
                )
        if not chips:
            return '<div class="trace-line">Nothing specific yet — we’ll ask a quick question.</div>'
        return '<div class="trace-chips">' + "".join(chips) + "</div>"
    if name == "Check we have enough":
        if output.get("question"):
            q = safe_text(output.get("question"))
            opts = [safe_text(o) for o in (output.get("options") or []) if safe_text(o)]
            rows = [f'<div class="trace-line">We asked: <em>“{escape(q)}”</em></div>']
            if opts:
                rows.append(
                    '<div class="trace-chips">'
                    + "".join(_chip_line(o) for o in opts)
                    + "</div>"
                )
            return "".join(rows)
        return '<div class="trace-line">Enough detail to go on — moving to the books.</div>'
    if name == "Rate every book":
        # The trace keeps only the strongest few candidates so the explanation is
        # readable even when the model scored a long shortlist.
        rows = []
        for s in output.get("scored", [])[:6]:
            title = safe_text(s.get("title")) or f"Book #{safe_text(s.get('book_id'))}"
            strength = _match_strength(s.get("relevance"))
            reason = safe_text(s.get("reason"))
            flags = [safe_text(f) for f in (s.get("flags") or []) if safe_text(f)]
            line = (
                f'<span class="trace-strength">{escape(strength)}</span> '
                f'<span class="trace-book">{escape(title)}</span>'
            )
            if reason:
                line += f' — {escape(reason)}'
            if flags:
                line += ' <span class="trace-flag">set aside: '
                line += escape(", ".join(f.replace("avoid:", "") for f in flags))
                line += "</span>"
            rows.append(f'<div class="trace-line">{line}</div>')
        head = (
            f'We rated {output.get("count", len(rows))} books — '
            f'showing the {len(rows)} strongest:'
        )
        return f'<div class="trace-line trace-head">{escape(head)}</div>' + "".join(rows)
    if name == "Pick the final list":
        rows = []
        for i, p in enumerate(output.get("picks", []), start=1):
            title = safe_text(p.get("title"), "Untitled")
            desc = safe_text(p.get("description"))
            why = safe_text(p.get("explanation"))
            line = (
                f'<span class="trace-rank">{i}</span> '
                f'<span class="trace-book">{escape(title)}</span>'
            )
            if desc:
                line += f' — {escape(desc)}'
            if why:
                line += f' <span class="trace-flag">Why: {escape(why)}</span>'
            rows.append(f'<div class="trace-line">{line}</div>')
        return "".join(rows)
    return ""


# Map the pipeline's internal stage names to reader-friendly step titles.
TRACE_STEP_NAMES = {
    "Intent": "Understand your request",
    "Clarify": "Check we have enough",
    "Scoring": "Rate every book",
    "Re-rank": "Pick the final list",
}


def render_reasoning_trace(trace) -> str:
    """A collapsible step-by-step explanation of how the shortlist was built.

    `trace` is a list of asdict() rag_pipeline.StageTrace dicts. Native HTML
    <details>, so it works inside st.markdown with no JS. Returns "" when absent.
    Written for a non-technical reader: numbered steps with plain-language notes.
    """
    if not trace:
        return ""
    stages = []
    for i, stage in enumerate(trace, start=1):
        raw_name = safe_text(stage.get("name"), "Stage")
        name = TRACE_STEP_NAMES.get(raw_name, raw_name)
        note = escape(safe_text(stage.get("note")))
        body = _trace_output_summary(name, stage.get("output") or {})
        body_html = f'<div class="trace-body">{body}</div>' if body else ""
        stages.append(
            f'<div class="trace-stage">'
            f'<div class="trace-stage-head">'
            f'<span class="trace-step-no">{i}</span>'
            f'<span class="trace-stage-name">{escape(name)}</span>'
            f'</div>'
            f'<div class="trace-note">{note}</div>'
            f'{body_html}'
            f'</div>'
        )
    count = len(trace)
    return (
        f'<details class="reasoning-trace">'
        f'<summary>How we chose these · {count} steps</summary>'
        f'<div class="trace-intro">A quick look at how BookRec went from your '
        f'message to this shortlist.</div>'
        f'<div class="trace-stages">' + "".join(stages) + '</div>'
        f'</details>'
    )


def render_clarify_message(message) -> str:
    """Build the HTML for an assistant turn that asks a clarifying question.

    Shows the question as the lead plus the intent chips for what we already
    understood. The tappable options are rendered as real Streamlit buttons in
    the composer area, not here.
    """
    intent = message.get("intent") or {}
    question = escape(safe_text(message.get("question_text"), "Could you tell me a bit more?"))
    source = escape(safe_text(message.get("source"), ""))

    parts = [
        '<div class="msg msg-assistant">',
        '<div class="assistant-avatar">B</div>',
        '<div class="assistant-body">',
        '<div class="assistant-name">BookRec</div>',
        f'<div class="assistant-lead">{question}</div>',
        render_intent_chips(intent),
        '<div class="clarify-hint">Tap one of the suggested answers, type your own, '
        'or skip straight to recommendations.</div>',
        render_reasoning_trace(message.get("trace")),
    ]
    if source:
        parts.append(f'<div class="assistant-meta">Source: {source}</div>')
    parts.append("</div>")  # assistant-body
    parts.append("</div>")  # msg
    return "\n".join(parts)


def _goodreads_url(cover_url, title, authors):
    """Best-effort link to the book's Goodreads page.

    The Goodreads cover filename is the goodreads_book_id (e.g.
    ``.../books/1447303603s/2767052.jpg`` -> 2767052), which gives a direct book
    page. When the cover is a generic placeholder (no numeric id), fall back to a
    Goodreads search by title + author so the link always works.
    """
    match = re.search(r"/(\d+)\.[a-zA-Z]+$", cover_url or "")
    if match:
        return f"https://www.goodreads.com/book/show/{match.group(1)}"
    query = urllib.parse.quote_plus(f"{title} {authors}".strip())
    return f"https://www.goodreads.com/search?q={query}"


@st.cache_data
def book_media_lookup(source, _books):
    """Map book_id -> {"cover": <url or "">, "url": <goodreads link>}.

    Lets the chat pick cards show the same cover thumbnail (from the catalog's
    image link) and link out to Goodreads, working from just the pick's book_id.

    Cached by `source` (a cheap string key); the catalog is passed as `_books`
    (underscore -> excluded from the cache key) so Streamlit never re-hashes the
    full ~10k-row frame on every rerun.
    """
    books = _books
    cover_col = next(
        (c for c in ("small_image_url", "image_url") if c in books.columns), None
    )
    out = {}
    for row in books.itertuples(index=False):
        cover = safe_text(getattr(row, cover_col, "")) if cover_col else ""
        out[getattr(row, "book_id")] = {
            "cover": cover,
            "url": _goodreads_url(
                cover,
                safe_text(getattr(row, "title", "")),
                safe_text(getattr(row, "authors", "")),
            ),
        }
    return out


def render_assistant_message(message, book_meta=None) -> str:
    """Build the HTML for one assistant turn (a re-ranked shortlist)."""
    if message.get("kind") == "clarify":
        return render_clarify_message(message)
    book_meta = book_meta or {}
    picks = message.get("picks", [])
    intent = message.get("intent") or {}
    lead_text = safe_text(intent.get("summary"))
    if not lead_text:
        # Backward-compatible path for older stored messages that predate the
        # structured Intent summary.
        pref = safe_text(message.get("pref"), "your request")
        verb = "Refined the shortlist" if message.get("refine") else "Here is your shortlist"
        lead_text = f"{verb} for “{pref}”."
    lead = escape(lead_text)
    source = escape(safe_text(message.get("source"), ""))

    parts = [
        '<div class="msg msg-assistant">',
        '<div class="assistant-avatar">B</div>',
        '<div class="assistant-body">',
        '<div class="assistant-name">BookRec</div>',
        f'<div class="assistant-lead">{lead}</div>',
        render_intent_chips(intent),
        '<div class="pick-list">',
    ]
    if picks:
        for rank, pick in enumerate(picks, start=1):
            title = escape(safe_text(pick.title, "Untitled"))
            authors = escape(safe_text(pick.authors, "Unknown author"))
            why = escape(safe_text(pick.explanation, "Ranked for this preference."))
            # description is the NEW contract field — one neutral sentence about
            # the book itself. May be empty for older/edge picks, so only render
            # the line when it is populated.
            desc = safe_text(getattr(pick, "description", ""))
            desc_html = (
                f'<div class="pick-desc">{escape(desc)}</div>' if desc else ""
            )
            # Cover thumbnail + Goodreads link, looked up by the pick's book_id.
            media = book_meta.get(getattr(pick, "book_id", None), {})
            cover = safe_text(media.get("cover", ""))
            link = safe_text(media.get("url", ""))
            if cover.startswith("http"):
                cover_inner = (
                    f'<img class="pick-cover-img" src="{escape(cover)}" alt="" '
                    f'loading="lazy" referrerpolicy="no-referrer">'
                )
            else:
                cover_inner = (
                    '<span class="pick-cover-img pick-cover--empty" '
                    'aria-hidden="true">📖</span>'
                )
            if link.startswith("http"):
                cover_html = (
                    f'<a class="pick-cover" href="{escape(link)}" target="_blank" '
                    f'rel="noopener noreferrer">{cover_inner}</a>'
                )
                title_html = (
                    f'<a class="pick-title-link" href="{escape(link)}" '
                    f'target="_blank" rel="noopener noreferrer">{title}</a>'
                )
            else:
                cover_html = f'<div class="pick-cover">{cover_inner}</div>'
                title_html = title
            parts.append(
                f'<article class="pick-card">'
                f'<div class="pick-rank">{rank}</div>'
                f'{cover_html}'
                f'<div class="pick-title">{title_html}'
                f'<span class="pick-author"> by {authors}</span></div>'
                f'{desc_html}'
                f'<div class="pick-why">'
                f'<span class="pick-why-label">Why #{rank}</span>{why}'
                f'</div>'
                f'</article>'
            )
    else:
        parts.append(
            '<div class="pick-empty">No candidates matched closely enough. '
            'Try regenerating candidates or loosening the filters above.</div>'
        )
    parts.append("</div>")  # pick-list
    parts.append(render_reasoning_trace(message.get("trace")))
    if source:
        parts.append(f'<div class="assistant-meta">Source: {source}</div>')
    parts.append("</div>")  # assistant-body
    parts.append("</div>")  # msg
    return "\n".join(parts)


def render_chat_thread(messages, pending: bool = False, book_meta=None) -> None:
    """Render the full conversation as alternating user / assistant turns."""
    parts = ['<div class="chat-thread">']
    for message in messages:
        if message["role"] == "user":
            parts.append(
                '<div class="msg msg-user">'
                f'<div class="bubble-user">{escape(message["content"])}</div>'
                '</div>'
            )
        else:
            parts.append(render_assistant_message(message, book_meta))
    if pending:
        stage_pills = "".join(
            f'<span class="stage-pill"><span class="stage-pill-dot"></span>{label}</span>'
            for label in ("Reading your request", "Rating the books", "Choosing the best")
        )
        parts.append(
            '<div class="msg msg-assistant">'
            '<div class="assistant-avatar">B</div>'
            '<div class="assistant-body">'
            '<div class="assistant-name">BookRec</div>'
            '<div class="assistant-lead">Working through your request…</div>'
            f'<div class="stage-progress">{stage_pills}</div>'
            '<div class="thinking"><span></span><span></span><span></span></div>'
            '</div>'
            '</div>'
        )
    parts.append("</div>")
    st.markdown("\n".join(parts), unsafe_allow_html=True)


@st.cache_data
def load_data(source: str):
    # Data frames are pure inputs to the rest of the app, so cache by selected
    # source and reuse across reruns triggered by UI interactions.
    # Absolute path so the app works regardless of the working directory — on
    # hosted platforms (e.g. Streamlit Community Cloud) the CWD is the repo root,
    # not this folder, so a bare "data" would not resolve.
    return load(os.path.join(_APP_DIR, "data"))


@st.cache_data
def auto_reader(source, _ratings):
    """Pick a sensible default reader for UBCF: the most active rater.

    Reader selection is exposed in the app, and the automatic choice defaults to
    the reader with the richest history so UBCF has the most signal. Cached by
    `source` so the value_counts() over ~165k ratings runs once, not every rerun.
    """
    ratings = _ratings
    return ratings["user_id"].value_counts().idxmax()


# Upper bound on how many reader ids the picker lists. The assignment catalog has
# ~1.2k users (all shown); the cap only guards against an unwieldy dropdown on a
# much larger dataset, and when it bites we say so in the help text rather than
# truncating silently.
READER_OPTIONS_MAX = 2000


@st.cache_data
def reader_options(source, _ratings):
    """(sorted user ids for the picker, truncated?) — cached by `source`.

    Sorted numerically so a specific id is easy to find; the value_counts/sort
    runs once per source instead of on every rerun.
    """
    ids = sorted(_ratings["user_id"].unique().tolist())
    return ids[:READER_OPTIONS_MAX], len(ids) > READER_OPTIONS_MAX


# An author needs at least this many books in the catalog to appear in the
# filter. With fewer titles, selecting that author leaves a candidate pool too
# thin for the collaborative filter to rank meaningfully, so we hide them and
# show only the well-represented (popular) authors.
MIN_AUTHOR_BOOKS = 5

# Cap the filter to this many of the most popular qualifying authors, so the
# dropdown stays a short, recognizable shortlist rather than a long tail.
TOP_AUTHORS = 50


@st.cache_data
def author_options(source, _books):
    """The TOP_AUTHORS most popular authors with enough books to seed a candidate list.

    Keeps authors with at least MIN_AUTHOR_BOOKS titles in the catalog, ranks
    them by total ratings (most popular first), and returns the top TOP_AUTHORS.
    Authors with only a handful of titles are dropped — selecting one would yield
    a candidate pool too small to re-rank. Falls back to every named author when
    nothing clears the threshold (e.g. a tiny catalog).

    Cached by `source`; `_books` is excluded from the cache key (underscore) so
    the frame isn't re-hashed every rerun.
    """
    books = _books
    if "authors" not in books.columns:
        return []
    exploded = books.assign(
        _author=books["authors"].fillna("").str.split(",")
    ).explode("_author")
    exploded["_author"] = exploded["_author"].str.strip()
    exploded = exploded[
        (exploded["_author"] != "") & (exploded["_author"] != "Unknown")
    ]
    if exploded.empty:
        return []
    exploded["_rc"] = pd.to_numeric(
        exploded.get("ratings_count", 0), errors="coerce"
    ).fillna(0)
    grouped = exploded.groupby("_author")
    stats = pd.DataFrame(
        {"n_books": grouped.size(), "popularity": grouped["_rc"].sum()}
    )
    popular = stats[stats["n_books"] >= MIN_AUTHOR_BOOKS]
    if popular.empty:
        # Threshold filtered everyone out (tiny catalog) — keep all named authors.
        return sorted(stats.index)
    popular = popular.sort_values(
        ["popularity", "n_books"], ascending=[False, False]
    )
    return list(popular.index[:TOP_AUTHORS])


# A decade needs at least this many books to stand alone as a filter option;
# the run of sparser early decades is collapsed into one "Before <cutoff>s" bin
# so the UBCF candidate pool for any selected period is never trivially small.
MIN_DECADE_BOOKS = 50


@st.cache_data
def decade_options(source, _books):
    """Decade labels for the filter — individual recent decades, sparse old ones binned.

    Decades with enough books (>= MIN_DECADE_BOOKS) are listed individually
    (e.g. "1990s"). The leading run of sparse early decades is collapsed into a
    single "Before <cutoff>s" option (cutoff = the first dense decade), so any
    selected period always has enough candidates for the collaborative filter.
    Falls back to listing every decade when there is no meaningful split (e.g.
    a tiny catalog). Cached by `source`; `_books` excluded from key.
    """
    books = _books
    years = pd.to_numeric(
        books.get("original_publication_year", pd.Series(dtype=float)),
        errors="coerce",
    ).dropna()
    years = years[years > 0]
    if years.empty:
        return []
    counts = (years // 10 * 10).astype(int).value_counts()
    decades = sorted(counts.index)
    dense = [d for d in decades if counts[d] >= MIN_DECADE_BOOKS]
    if len(dense) < 2 or dense[0] == decades[0]:
        # Nothing sparse to collapse — list every decade individually.
        return [f"{d}s" for d in decades]
    cutoff = dense[0]
    return [f"Before {cutoff}s"] + [f"{d}s" for d in decades if d >= cutoff]


def filter_book_ids(books, authors=None, decades=None):
    """Restrict the catalog by author/decade.

    Returns None when nothing is selected (a true no-op for the caller). When any
    group is active, builds a vectorized boolean mask: AND across active groups,
    OR within a group (a book matches a group if ANY of its split values is
    selected). Returns set(book_id) of the surviving rows.
    """
    authors = authors or []
    decades = decades or []
    if not authors and not decades:
        return None

    mask = pd.Series(True, index=books.index)

    if authors:
        wanted = set(authors)
        author_split = books.get("authors", pd.Series("", index=books.index)).fillna("")
        author_mask = author_split.apply(
            lambda cell: any(
                part.strip() in wanted for part in str(cell).split(",")
            )
        )
        mask &= author_mask

    if decades:
        # Decade labels are either an exact decade ("1990s") or the collapsed
        # "Before <cutoff>s" bin; handle both (and any combination of them).
        exact_decades = set()
        before_cutoff = None
        for d in decades:
            label = str(d)
            if label.startswith("Before "):
                try:
                    before_cutoff = int(label.replace("Before ", "").rstrip("s"))
                except ValueError:
                    pass
            else:
                try:
                    exact_decades.add(int(label.rstrip("s")))
                except ValueError:
                    pass
        years = pd.to_numeric(
            books.get("original_publication_year", pd.Series(0, index=books.index)),
            errors="coerce",
        ).fillna(0)
        decade_mask = (years // 10 * 10).astype(int).isin(exact_decades) & (years > 0)
        if before_cutoff is not None:
            decade_mask = decade_mask | ((years > 0) & (years < before_cutoff))
        mask &= decade_mask

    return set(books.loc[mask, "book_id"])


@st.cache_resource
def build_model(source: str, kind: str, k: int = DEFAULT_K, sim_name: str | None = None):
    # Model objects are heavier and mutable, so cache as resources rather than
    # serializing them through st.cache_data. scikit-surprise is required (the
    # main flow stops earlier with a clear message if it is unavailable).
    ratings, _ = load_data(source)
    return CFModel(kind=kind, k=k, sim_name=sim_name).fit(ratings)


@st.cache_resource
def build_content_model(source: str):
    """TF-IDF content model over book metadata, used to ground the re-ranker:
    for each CF candidate we find the reader's own highly-rated book it most
    resembles. Cached so it's fit once per data source."""
    _, books = load_data(source)

    b = books.copy()
    if "tags" not in b.columns:  # the assignment catalog has no shelf tags
        b["tags"] = ""
    return ContentModel().fit(b)


def compute_grounding(source, uid, ratings, books, cand_ids):
    """Map each candidate book_id -> the title of the reader's most content-similar
    favorite, for grounding the LLM re-ranker's explanations. Never raises."""
    try:
        cm = build_content_model(source)
        liked = ratings.loc[
            (ratings["user_id"] == uid) & (ratings["rating"] >= 4.0), "book_id"
        ].tolist()
        title_of = dict(zip(books["book_id"], books["title"]))
        return {
            bid: title_of.get(lid, "")
            for bid, (lid, _sim) in cm.nearest_examples(cand_ids, liked).items()
            if title_of.get(lid)
        }
    except Exception:
        return {}


class ScoreAdapter:
    """Wrap any model so recommend_top_n can call .score(...)."""

    def __init__(self, model):
        self.model = model

    def score(self, user_id, user_ratings, candidate_ids):
        return self.model.predict_for_user(user_id, candidate_ids)


def submit_chat_message(prompt: str, top_k: int, skip_clarify: bool = False) -> None:
    """Append the user's turn and queue an assistant reply for the next run.

    ``skip_clarify`` powers the "Just recommend something" escape hatch — it tells
    the DAG to bypass the clarifying-question gate and rank immediately.
    """
    messages = st.session_state.setdefault("chat_messages", [])
    messages.append({"role": "user", "content": prompt})
    st.session_state["chat_pending"] = {"top_k": top_k, "skip_clarify": skip_clarify}
    st.session_state["clear_chat_input"] = True


def resolve_pending_chat(books) -> None:
    """Run the conversation-aware RAG re-rank for the queued user turn.

    The DAG may pause at its clarify gate and return a question instead of picks;
    we append either a ``kind:"clarify"`` or a ``kind:"recommend"`` assistant turn.
    """
    pending = st.session_state.get("chat_pending")
    if not pending:
        return
    st.session_state["chat_pending"] = None  # clear early so we never re-enter

    messages = st.session_state.get("chat_messages", [])
    recs = st.session_state.get("cf_recs")
    user_turns = [m["content"] for m in messages if m["role"] == "user"]
    if recs is None or not user_turns:
        return

    # How many clarifying questions we've already asked this conversation; the
    # gate stops asking once it hits the pipeline's max-rounds cap.
    rounds_asked = sum(1 for m in messages if m.get("kind") == "clarify")

    cands = candidates_from_recs(recs, books)
    # Attach content-similarity grounding so the re-ranker can tie a pick to a
    # book this reader already loved ("in the spirit of X").
    grounding = st.session_state.get("grounding") or {}
    for c in cands:
        sim = grounding.get(c["book_id"])
        if sim:
            c["similar_to"] = sim
    try:
        result = run_pipeline(
            user_turns,
            cands,
            top_k=pending["top_k"],
            rounds_asked=rounds_asked,
            skip_clarify=pending.get("skip_clarify", False),
        )
    except Exception as exc:  # noqa: BLE001 — last-resort UI guard
        # LLM-only pipeline: a raise here means the Gemini call failed (commonly
        # the free-tier rate limit) after retries. Keep the app alive and show an
        # honest turn rather than a traceback (queued turn already cleared above).
        messages.append(
            {
                "role": "assistant",
                "kind": "recommend",
                "picks": [],
                "source": "error",
                "used_llm": False,
                "refine": len(user_turns) > 1,
                "pref": user_turns[-1],
                "intent": {
                    "summary": "The AI re-ranker is temporarily unavailable "
                    "(often the Gemini rate limit). Please wait a moment and try "
                    "again."
                },
                "trace": [],
            }
        )
        return

    if result.question is not None:
        messages.append(
            {
                "role": "assistant",
                "kind": "clarify",
                "question_text": result.question.prompt,
                "options": list(result.question.options),
                "source": result.source,
                "used_llm": result.used_llm,
                "intent": asdict(result.intent),
                "trace": [asdict(stage) for stage in result.trace],
            }
        )
        return

    messages.append(
        {
            "role": "assistant",
            "kind": "recommend",
            "picks": result.picks,
            "source": result.source,
            "used_llm": result.used_llm,
            "refine": len(user_turns) > 1,
            "pref": user_turns[-1],
            "intent": asdict(result.intent),
            "trace": [asdict(stage) for stage in result.trace],
        }
    )


inject_css()

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    # Either Gemini SDK enables the live LLM layer: the new google-genai
    # (preferred — structured output) or the legacy google-generativeai.
    HAVE_GEMINI_SDK = _sdk_available()

HAVE_GEMINI_KEY = bool(os.environ.get("GEMINI_API_KEY"))
# The chat re-ranker is LLM-only — both a key and the SDK are required for it.
HAVE_LLM = HAVE_GEMINI_KEY and HAVE_GEMINI_SDK

# Compact runtime health badge for the fixed nav.
if not HAVE_SURPRISE:
    nav_status = "Engine unavailable"
elif HAVE_LLM:
    nav_status = "Gemini ready"
elif HAVE_GEMINI_KEY:
    nav_status = "SDK missing"
else:
    nav_status = "Gemini key required"

st.markdown(
    '<nav class="floating-nav" aria-label="BookRec sections">'
    '<div class="floating-nav-inner">'
    '<a class="top-brand" href="#filtering">'
    '<span class="brand-mark">B</span>'
    '<span>'
    '<span class="brand-wordmark">BookRec</span>'
    '<span class="brand-caption">Personalized recommendations</span>'
    '</span>'
    '</a>'
    '<div class="top-links">'
    '<a class="nav-pill" href="#filtering">Filter</a>'
    '<a class="nav-pill" href="#candidates">Candidates</a>'
    '<a class="nav-pill" href="#personalize">Chat</a>'
    '</div>'
    f'<span class="nav-status"><span class="status-dot"></span>{escape(nav_status)}</span>'
    '<a class="nav-cta" href="#personalize">Personalize</a>'
    '</div>'
    '</nav>',
    unsafe_allow_html=True,
)

def book_markup(extra_classes: str, n_leaves: int = 4) -> str:
    """A CSS-only 3D book whose pages continuously riffle (fallback visual)."""
    leaves = "".join(f'<div class="leaf leaf-{i}"></div>' for i in range(1, n_leaves + 1))
    return (
        f'<div class="book {extra_classes}" aria-hidden="true">'
        '<div class="book-3d">'
        '<div class="book-cover"></div>'
        '<div class="page-static page-left"></div>'
        '<div class="page-static page-right"></div>'
        f'{leaves}'
        '<div class="spine"></div>'
        '</div>'
        '</div>'
    )


@st.cache_data
def _encode_asset(path: str, mtime: float) -> str:
    # mtime is unused in the body but is part of the cache key, so editing the
    # file on disk (new mtime) busts the cache. Do NOT prefix it with "_" —
    # st.cache_data ignores underscore-prefixed args when hashing the key.
    with open(path, "rb") as handle:
        return base64.b64encode(handle.read()).decode("ascii")


def asset_data_uri(filename: str):
    """Base64 data URI for an asset in ./assets, or None if it is missing.

    Streamlit can't serve local file paths inside raw HTML, so decorative art
    is embedded directly. Cached by path + mtime so regenerated files are
    picked up without restarting the server.
    """
    path = os.path.join(_APP_DIR, "assets", filename)
    if not os.path.exists(path):
        return None
    return f"data:image/webp;base64,{_encode_asset(path, os.path.getmtime(path))}"


def art_panel(uri, side: str, kind: str, fallback_classes: str) -> str:
    """Side artifact image, falling back to the CSS book if the asset is absent."""
    if uri:
        return (
            f'<div class="art-panel art-{side} art-{kind}" aria-hidden="true">'
            f'<img src="{uri}" alt=""></div>'
        )
    return book_markup(fallback_classes, n_leaves=3)


spiral_uri = asset_data_uri("spiral.webp")
nodes_uri = asset_data_uri("nodes.webp")
book_uri = asset_data_uri("book.webp")

# Ambient artifacts framing the page (fixed; wide screens only).
st.markdown(
    art_panel(spiral_uri, "left", "spiral", "side-book side-left book-side")
    + art_panel(nodes_uri, "right", "nodes", "side-book side-right book-side"),
    unsafe_allow_html=True,
)

if book_uri:
    hero_visual = (
        '<div class="center-stage" aria-hidden="true">'
        f'<img class="center-book-img" src="{book_uri}" alt="">'
        '<div class="flip-layer">'
        '<div class="img-leaf leaf-a"></div>'
        '<div class="img-leaf leaf-b"></div>'
        '<div class="img-leaf leaf-c"></div>'
        '</div>'
        '</div>'
    )
else:
    hero_visual = book_markup("book-hero")

st.markdown('<div id="filtering"></div>', unsafe_allow_html=True)
st.markdown(
    '<header class="cine-hero">'
    + hero_visual
    + '<div class="page-kicker">BookRec filtering workspace</div>'
    '<h1 class="page-title">Build the shortlist first. Personalize it at the bottom.</h1>'
    '<p class="page-copy">Select the reader, tune the collaborative filter, review the candidate books, then scroll into a chat-style personalization pass.</p>'
    '</header>',
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.markdown('<div class="section-title">Filtering</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">Select the reader to recommend for, then optionally filter by author and decade to shape the candidate pool the chat assistant will re-rank.</div>',
        unsafe_allow_html=True,
    )

    # The base view shows only the author + decade filters. They depend on the
    # loaded catalog (which depends on the data source, an advanced setting), so
    # we reserve their row here and fill it after the data has loaded below.
    base_filters = st.container()

    # Collaborative filtering needs scikit-surprise. It's a hard requirement now
    # (only UBCF/IBCF are supported — no non-personalized fallback), so stop with
    # a clear message rather than erroring deeper if the wheel didn't install.
    if not HAVE_SURPRISE:
        st.error(
            "This app needs **scikit-surprise** for collaborative filtering, but "
            "it isn't installed. Install it (`pip install scikit-surprise`, with "
            "NumPy < 2.0) and reload."
        )
        st.stop()

    # Model and tuning are fixed to the validated optimal configuration — no UI
    # knobs. We ship UBCF (user-based, pearson_baseline, k=25): the best
    # MEMORY-LIGHT model in the corrected Top-N audit and the top user-based
    # config on the held-out test (Precision@10 0.6953 / F1 0.745; see
    # notebooks/precision_at_k_corrected.ipynb cell 46 and scripts/run_cf_bakeoff.py).
    # The chat re-ranks these candidates on top.
    #
    # Why not IBCF: item-based pearson_baseline scored marginally higher offline
    # (F1 0.746, best RMSE) but builds a ~9k x 9k item-item similarity matrix —
    # measured ~815MB peak RSS vs ~177MB for this UBCF config — which OOM-kills
    # Streamlit Community Cloud's free tier on the first candidate build. The
    # ranking-quality gap is ~0.001 F1 (imperceptible to the top-N the app shows).
    # If you deploy on a host with more RAM and want the absolute-best metrics,
    # switch the two lines below to:  cf_kind = "ibcf";  k_neighbors = 9.
    source = SOURCE_REAL
    cf_kind = "ubcf"
    cf_sim = BEST_SIM
    k_neighbors = DEFAULT_UBCF_K
    top_n = 10
    min_ratings = DEFAULT_MIN_RATINGS

    with st.spinner("Loading catalog..."):
        ratings, books = load_data(source)

    # UBCF needs a user; default to the most active reader unless overridden.
    # These helpers are cached by `source` (cheap key) and take the frame as an
    # underscore arg, so they don't re-hash the full catalog on every rerun.
    auto_uid = auto_reader(source, ratings)
    author_opts = author_options(source, books)
    decade_opts = decade_options(source, books)

    # Reader selection is a primary control (the assignment asks the app to let a
    # user be selected) — it sits up front with the author/decade filters.
    with base_filters:
        reader_col, author_col, decade_col = st.columns(3)
        with reader_col:
            reader_ids, readers_truncated = reader_options(source, ratings)
            reader_help = (
                "The user the collaborative filter personalizes for. Auto uses "
                "the most active reader; the chat refines on top of these "
                "candidates."
            )
            if readers_truncated:
                reader_help += (
                    f" Showing the first {READER_OPTIONS_MAX:,} reader ids."
                )
            reader_choice = st.selectbox(
                "Reader (user to recommend for)",
                ["Auto (most active)"] + reader_ids,
                help=reader_help,
                key="reader_override",
            )
        with author_col:
            sel_authors = st.multiselect(
                "Authors",
                author_opts,
                key="filt_authors",
                help="The 50 most popular authors with enough books in the "
                     "catalog to seed a candidate list, ordered by popularity. "
                     "Type to search.",
            )
        with decade_col:
            sel_decades = st.multiselect("Decades", decade_opts, key="filt_decades")

    uid = auto_uid if str(reader_choice).startswith("Auto") else reader_choice
    allowed_book_ids = filter_book_ids(books, sel_authors, sel_decades)

    # Model/tuning are now constant, so only the reader + filters drive a rebuild.
    current_config = (
        uid,
        tuple(sorted(sel_authors)),
        tuple(sorted(sel_decades)),
    )
    if st.session_state.get("rec_config") != current_config:
        st.session_state["cf_recs"] = None
        st.session_state["chat_messages"] = []
        st.session_state["chat_pending"] = None
        st.session_state["grounding"] = {}

    filter_pills = ""
    if sel_authors:
        filter_pills += f'<span class="pill">{len(sel_authors)} author(s)</span>'
    if sel_decades:
        filter_pills += (
            f'<span class="pill">{escape(", ".join(sel_decades))}</span>'
        )
    if filter_pills:
        st.markdown(
            f'<div class="pill-row">{filter_pills}</div>',
            unsafe_allow_html=True,
        )

    if st.button("Generate filtered candidates", type="primary", width="stretch"):
        with st.spinner("Scoring the catalog..."):
            model = build_model(source, cf_kind, k=k_neighbors, sim_name=cf_sim)
            scorer = ScoreAdapter(model)
            st.session_state["cf_recs"] = recommend_top_n(
                uid,
                scorer,
                ratings,
                books,
                top_n=top_n,
                min_ratings=min_ratings,
                allowed_book_ids=allowed_book_ids,
            )
            st.session_state["rec_config"] = current_config
            st.session_state["chat_messages"] = []
            st.session_state["chat_pending"] = None
            # Precompute content-similarity grounding for the chat re-ranker.
            st.session_state["grounding"] = compute_grounding(
                source, uid, ratings, books,
                list(st.session_state["cf_recs"]["book_id"]),
            )
        if allowed_book_ids is not None and st.session_state["cf_recs"].empty:
            st.warning(
                "Your filters removed every candidate — relax a filter or lower "
                "the minimum ratings."
            )

st.markdown('<div id="candidates" class="section"></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-title">Candidate list</div>'
    '<div class="section-copy">This is the model-produced shortlist. The chat section below reorders these books without inventing new titles.</div>',
    unsafe_allow_html=True,
)

recs = st.session_state.get("cf_recs")
if recs is None:
    with st.container(border=True):
        st.markdown(
            '<div class="empty-state"><div>'
            '<div class="empty-title">No candidates yet</div>'
            '<div>Use the filters above to generate a ranked list.</div>'
            '</div></div>',
            unsafe_allow_html=True,
        )
else:
    st.markdown(
        f'<div class="section-copy">{escape(model_label(cf_kind))} scored '
        f'{len(recs):,} books matching your filters.</div>',
        unsafe_allow_html=True,
    )
    render_recommendation_cards(recs, books)
    with st.expander("Open as table"):
        st.dataframe(
            recs[["book_id", "title", "authors", "score"]],
            width="stretch",
            hide_index=True,
        )

st.markdown('<div id="personalize" class="section"></div>', unsafe_allow_html=True)
with st.container(key="chat_stage"):
    # Clear the composer on the run after a message is sent (Streamlit only lets
    # us reset a widget's value before the widget is instantiated).
    if st.session_state.pop("clear_chat_input", False):
        st.session_state["chat_pref"] = ""

    messages = st.session_state.setdefault("chat_messages", [])
    pending = bool(st.session_state.get("chat_pending"))
    has_recs = st.session_state.get("cf_recs") is not None
    refining = bool(messages)
    # The DAG paused to ask a question if the last turn is a clarify message and
    # we're not mid-run; the composer then shows the question's quick-reply chips.
    last_message = messages[-1] if messages else None
    awaiting_clarify = (
        not pending
        and last_message is not None
        and last_message.get("role") == "assistant"
        and last_message.get("kind") == "clarify"
    )

    if refining:
        render_chat_thread(
            messages, pending=pending, book_meta=book_media_lookup(source, books)
        )
    else:
        st.markdown(
            '<div class="chat-hero">'
            '<div class="chat-title">Ready when you are.</div>'
            '<div class="chat-subtitle">Ask BookRec for a mood, genre, theme, or '
            'reading goal — then keep refining. It personalizes the filtered '
            'candidates above and never invents books outside the shortlist.</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    if not has_recs:
        st.markdown(
            '<div class="chat-lock">Generate filtered candidates first so the chat '
            'can re-rank real books.</div>',
            unsafe_allow_html=True,
        )
    elif not HAVE_LLM:
        # The chat re-ranker is LLM-only; without Gemini there is no fallback.
        msg = (
            "Set a <strong>GEMINI_API_KEY</strong> to enable the AI re-ranker."
            if not HAVE_GEMINI_KEY
            else "The Gemini SDK isn’t installed — run "
                 "<code>pip install google-genai</code> to enable the AI re-ranker."
        )
        st.markdown(
            f'<div class="chat-lock">{msg} The chat is powered only by the live '
            'LLM — there is no rule-based fallback.</div>',
            unsafe_allow_html=True,
        )

    composer_disabled = pending or not has_recs or not HAVE_LLM

    # Picks count (Depth control). Read from session_state so it's available
    # BEFORE the selectbox is re-instantiated below — the clarify answer chips,
    # which now render above the text box, need it too.
    personalized_k = (
        5 if st.session_state.get("chat_depth", "Focused") == "Focused" else 8
    )

    # When the assistant paused to ask a clarifying question, show the tappable
    # answers RIGHT under the question (above the text box) with a clear label,
    # so how to reply is obvious. You can still type your own answer or skip.
    if awaiting_clarify:
        clarify_options = [
            o for o in (last_message.get("options") or []) if str(o).strip()
        ]
        with st.container(key="chat_clarify"):
            st.markdown(
                '<div class="clarify-answer-label">'
                'Tap an answer below — or type your own in the box and hit Send'
                '</div>',
                unsafe_allow_html=True,
            )
            if clarify_options:
                chip_cols = st.columns(len(clarify_options), gap="small")
                for idx, (col, option) in enumerate(zip(chip_cols, clarify_options)):
                    with col:
                        # Key by index, not option text, so duplicate option
                        # strings can never collide into a DuplicateWidgetID.
                        if st.button(
                            option,
                            key=f"clarify_opt_{idx}",
                            disabled=composer_disabled,
                            width="stretch",
                        ):
                            submit_chat_message(option, personalized_k)
                            st.rerun()
        skip_cols = st.columns(2, gap="small")
        with skip_cols[0]:
            if st.button(
                "↻ Start a new chat", key="new_chat_clarify",
                width="stretch", disabled=pending,
            ):
                st.session_state["chat_messages"] = []
                st.session_state["chat_pending"] = None
                st.session_state["clear_chat_input"] = True
                st.rerun()
        with skip_cols[1]:
            if st.button(
                "Just recommend something →",
                disabled=composer_disabled,
                width="stretch",
            ):
                submit_chat_message(
                    "Just recommend something", personalized_k, skip_clarify=True
                )
                st.rerun()

    with st.container(key="chat_composer"):
        chat_pref = st.text_area(
            "Personalization prompt",
            key="chat_pref",
            label_visibility="collapsed",
            placeholder=(
                "Type your answer, or refine — adjust tone, pace, setting, "
                "tropes to avoid, or steer it somewhere new..."
                if refining
                else "Ask for dark academia, a cozy mystery, fast-paced sci-fi, "
                "or whatever mood you're in..."
            ),
            height=92,
        )
        control_cols = st.columns(
            [0.64, 0.18, 0.18],
            gap="small",
            vertical_alignment="center",
        )
        with control_cols[1]:
            st.selectbox(
                "Depth",
                ["Focused", "Extended"],
                key="chat_depth",
                label_visibility="collapsed",
                help="Focused returns 5 picks; Extended returns 8.",
            )
        with control_cols[2]:
            send = st.button(
                "Send",
                type="primary",
                disabled=composer_disabled,
                width="stretch",
            )

    if send and chat_pref.strip():
        submit_chat_message(chat_pref.strip(), personalized_k)
        st.rerun()

    # Suggestion chips under the box: refine chips after a recommendation,
    # starter chips before the first turn. (Clarify answers render above the box.)
    if refining and not awaiting_clarify:
        refine_chips = [
            "Make them darker and moodier",
            "Lean more recent",
            "Lighter and funnier",
            "Surprise me with a wildcard",
        ]
        with st.container(key="chat_suggestions"):
            chip_cols = st.columns(len(refine_chips), gap="small")
            for col, chip in zip(chip_cols, refine_chips):
                with col:
                    if st.button(chip, disabled=composer_disabled, width="stretch"):
                        submit_chat_message(chip, personalized_k)
                        st.rerun()
        reset_cols = st.columns([0.62, 0.38])
        with reset_cols[1]:
            if st.button(
                "↻ Start a new chat", key="new_chat_refine",
                width="stretch", disabled=pending,
            ):
                st.session_state["chat_messages"] = []
                st.session_state["chat_pending"] = None
                st.session_state["clear_chat_input"] = True
                st.rerun()
    elif not refining:
        starter_chips = [
            "Something fast-paced and adventurous",
            "A thoughtful literary list with emotional depth",
            "Dark, mysterious books with strong atmosphere",
        ]
        with st.container(key="chat_suggestions"):
            chip_cols = st.columns(len(starter_chips), gap="medium")
            for col, chip in zip(chip_cols, starter_chips):
                with col:
                    if st.button(chip, disabled=composer_disabled, width="stretch"):
                        submit_chat_message(chip, personalized_k)
                        st.rerun()

    st.markdown(
        '<div class="chat-toolbar">BookRec only re-ranks the generated candidates '
        '— keep refining to steer tone, pace, or setting. It never invents books '
        'outside the shortlist.</div>',
        unsafe_allow_html=True,
    )

    # Resolve the queued turn last so the user's message + the animated "thinking"
    # dots paint first, then the re-rank runs and we rerun to show the picks.
    if st.session_state.get("chat_pending"):
        resolve_pending_chat(books)
        st.rerun()
