"""
LLM personalization layer (Project requirement — Week 4 material).

Takes the Top-N candidates from your best collaborative-filtering model and
asks an LLM to RE-RANK and personalize them to a user's stated preference
(a mood, a favorite genre, etc.), returning a short explanation per pick.

Key design rules from the assignment:
  * The LLM RE-RANKS the CF candidates — it must NOT invent new books.
    We enforce this by only accepting book_ids that were in the candidate set.
  * Book metadata (title, authors, year, average_rating) is passed as context.
  * The API key is read from the environment (GEMINI_API_KEY) — never hard-coded.
  * Gemini is the recommended provider (free tier); cite model+provider in your deck.

Graceful fallback: if no API key / SDK is available, a transparent heuristic
re-ranker runs instead, so the app and notebook always work (and you can
develop offline). The heuristic is clearly labeled as the fallback.

Cited model/provider (fill in what you actually use):
    provider = "Google", model = "gemini-2.0-flash"
"""
from __future__ import annotations

import os
import re
import json
from dataclasses import dataclass

DEFAULT_MODEL = "gemini-2.0-flash"


@dataclass
class RerankedPick:
    book_id: int
    title: str
    authors: str
    explanation: str


def _candidate_table(candidates):
    """candidates: list of dicts with book_id, title, authors, year, average_rating, cf_score.
    Returns a compact, numbered context string for the prompt."""
    lines = []
    for c in candidates:
        lines.append(
            f"[{c['book_id']}] \"{c['title']}\" by {c.get('authors','?')} "
            f"({c.get('year','?')}), avg_rating={c.get('average_rating','?')}, "
            f"cf_score={c.get('cf_score','?')}"
        )
    return "\n".join(lines)


def build_prompt(candidates, preference: str, top_k: int) -> str:
    """The re-ranking prompt. Constrains output to the candidate book_ids."""
    table = _candidate_table(candidates)
    ids = [c["book_id"] for c in candidates]
    return (
        "You are a book recommendation assistant. A collaborative-filtering model "
        "produced the following candidate books for a user. Your job is to RE-RANK "
        "these candidates to best match the user's stated preference, and explain each pick.\n\n"
        f"User preference: \"{preference}\"\n\n"
        f"Candidate books (you may ONLY choose from these, by their [book_id]):\n{table}\n\n"
        f"Instructions:\n"
        f"- Select and order the {top_k} best matches for the preference.\n"
        f"- Do NOT invent books. Use only book_ids from this list: {ids}.\n"
        f"- For each, give a one-sentence explanation tied to the preference and the book's metadata.\n"
        f"- Respond with STRICT JSON: a list of objects "
        f'{{"book_id": <int>, "explanation": "<str>"}} and nothing else.'
    )


def _have_gemini() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY")) and _sdk_available()


def _sdk_available() -> bool:
    try:
        import google.generativeai  # noqa: F401
        return True
    except Exception:
        return False


def _call_gemini(prompt: str, model: str) -> str:
    import google.generativeai as genai
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    resp = genai.GenerativeModel(model).generate_content(prompt)
    return resp.text


def _parse_json_list(text: str):
    """Pull the first JSON array out of the model's reply."""
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return []


def _heuristic_rerank(candidates, preference, top_k):
    """Transparent offline fallback: boost candidates whose title/authors contain
    words from the stated preference, tie-broken by CF score. Clearly NOT an LLM."""
    pref_terms = {w.lower() for w in re.findall(r"[a-zA-Z]+", preference)}
    scored = []
    for c in candidates:
        text = f"{c['title']} {c.get('authors','')}".lower()
        overlap = sum(1 for t in pref_terms if t in text)
        scored.append((overlap, c.get("cf_score", 0.0), c))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    picks = []
    for overlap, _, c in scored[:top_k]:
        why = (f"matches “{preference}” on its title/author"
               if overlap else f"top collaborative-filtering pick for this user")
        picks.append(RerankedPick(c["book_id"], c["title"], c.get("authors", ""), why))
    return picks


def rerank(candidates, preference: str, top_k: int = 5,
           model: str = DEFAULT_MODEL, force_fallback: bool = False):
    """Re-rank CF candidates by the user's preference.

    Parameters
    ----------
    candidates : list[dict] with keys book_id, title, authors, year,
                 average_rating, cf_score (the CF Top-N).
    preference : the user's stated mood/genre/etc.
    Returns (picks: list[RerankedPick], used_llm: bool).
    """
    by_id = {c["book_id"]: c for c in candidates}

    if force_fallback or not _have_gemini():
        return _heuristic_rerank(candidates, preference, top_k), False

    prompt = build_prompt(candidates, preference, top_k)
    try:
        raw = _call_gemini(prompt, model)
        parsed = _parse_json_list(raw)
    except Exception:
        return _heuristic_rerank(candidates, preference, top_k), False

    picks = []
    for item in parsed:
        bid = item.get("book_id")
        if bid in by_id:                          # enforce: only real candidates
            c = by_id[bid]
            picks.append(RerankedPick(bid, c["title"], c.get("authors", ""),
                                      item.get("explanation", "")))
        if len(picks) >= top_k:
            break
    if not picks:                                 # model returned nothing usable
        return _heuristic_rerank(candidates, preference, top_k), False
    return picks, True


def candidates_from_recs(recs_df, books):
    """Helper: turn a recommend_top_n() DataFrame into the candidate dicts this
    module expects, pulling metadata from the books frame."""
    meta = books.set_index("book_id")
    out = []
    for r in recs_df.itertuples(index=False):
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
