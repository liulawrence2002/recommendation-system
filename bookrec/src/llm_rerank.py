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
    provider = "Google", model = "gemini-2.5-flash-lite"
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from enum import Enum

try:  # Pydantic ships as a dependency of the google-genai SDK.
    from pydantic import BaseModel, Field
    _HAVE_PYDANTIC = True
except Exception:  # keep the heuristic path importable even without it
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
PERSONA = (
    "You are BookRec, a warm, sharp reader's-advisory librarian. You match books "
    "to a reader's stated mood, pace, genre, and themes. Every explanation you "
    "write must be specific to THAT book — name something concrete about its story, "
    "tone, author, era, or reception — and tie it back to the reader's request. "
    "Never reuse the same reason or phrasing twice in one list; if two books fit "
    "for the same reason, differentiate them. Keep each explanation to one natural "
    "sentence, friendly and free of jargon."
)


@dataclass
class RerankedPick:
    book_id: int
    title: str
    authors: str
    explanation: str          # WHY this book earned its rank — ties to the reader intent
    description: str = ""      # ONE neutral sentence describing what the book IS / is about


# --- Week-4 structured-output schema ------------------------------------------
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
        f"- For each, give a one-sentence neutral DESCRIPTION of what the book itself is "
        f"about, plus a one-sentence EXPLANATION of why it fits the preference and the "
        f"book's metadata.\n"
        f"- Make every EXPLANATION distinct and specific to that book (its story, tone, "
        f"author, era, or rating); never reuse the same reason or wording across picks.\n"
        f"- Respond with STRICT JSON: a list of objects "
        f'{{"book_id": <int>, "description": "<one sentence about the book>", '
        f'"explanation": "<one sentence on why it fits the preference>"}} and nothing else.'
    )


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


def _have_gemini() -> bool:
    # Both pieces are required: a key alone is not enough if neither Gemini SDK
    # is installed, and an SDK alone cannot make authenticated calls.
    return bool(os.environ.get("GEMINI_API_KEY")) and _sdk_available()


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
        return parsed.model_dump()
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

# A single transient 429 used to drop the whole turn to the rule-based fallback.
# The free tier's per-minute (RPM) burst limit clears in a second or two, so we
# retry a rate-limited call a couple of times with exponential backoff before
# giving up. Non-rate-limit errors (and a genuinely exhausted daily quota) still
# fall through fast to the heuristic.
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
    Raises on hard failure so callers drop to their heuristic fallback.
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


def _heuristic_rerank(candidates, preference, top_k):
    """Transparent offline fallback: boost candidates whose title/authors contain
    words from the stated preference, tie-broken by CF score. Clearly NOT an LLM."""
    # Preference terms are intentionally simple and inspectable so a fallback
    # ranking can be explained honestly in the UI.
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
        picks.append(RerankedPick(c["book_id"], c["title"], c.get("authors", ""), why,
                                  description=_synth_description(c)))
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
        parsed = _structured(prompt, RERANK_LIST_SCHEMA, model, is_list=True, system=PERSONA)
    except Exception:
        return _heuristic_rerank(candidates, preference, top_k), False

    picks = []
    for item in parsed:
        bid = item.get("book_id")
        if bid in by_id:                          # enforce: only real candidates
            c = by_id[bid]
            # Trust the model for prose, but always recover missing descriptions
            # from local metadata so UI cards stay complete.
            desc = str(item.get("description", "")).strip() or _synth_description(c)
            picks.append(RerankedPick(bid, c["title"], c.get("authors", ""),
                                      item.get("explanation", ""), description=desc))
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
