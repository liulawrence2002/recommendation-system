"""
Sequential DAG for the conversational RAG re-ranker (Project 2).

Where ``llm_rerank.rerank`` does the whole personalization in ONE LLM call, this
module chains three dependent stages so each prompt is grounded in the output of
the previous one — a small sequential DAG:

    user turns
        │
        ▼
    [A] extract_intent      → structured Intent (mood, genres, pace, avoid, ...)
        │ Intent
        ▼
    [B] score_candidates    → per-candidate relevance + reason, grounded in Intent
        │ ScoredCandidate[]
        ▼
    [C] rerank_with_intent  → ordered RerankedPick[], explanations tied to Intent

Design rules carried over from ``llm_rerank``:
  * The LLM only ever RE-RANKS / scores the CF candidates — it cannot invent
    books. Stages B and C re-validate every ``book_id`` against the candidate set.
  * The API key is read from the environment (``GEMINI_API_KEY``) — never hard-coded.
  * Every stage has a transparent, deterministic heuristic fallback, so the whole
    DAG runs end-to-end offline. Stages fall back independently: a live Stage A can
    feed a heuristic Stage B and vice versa, because stages exchange structured
    objects rather than raw model text.

Each LLM stage uses ``llm_rerank._structured`` — Week-4 structured output
(Gemini ``response_schema``) — so the model returns typed JSON instead of free
text we have to regex-parse. This module reuses helpers from ``llm_rerank``
(``_structured``, ``_have_gemini``, ``_candidate_table``, ``RerankedPick``,
``DEFAULT_MODEL``) and adds no new public surface there. ``llm_rerank.rerank``
is left intact for the notebook.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

from . import llm_rerank
from .llm_rerank import DEFAULT_MODEL, RerankedPick

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

# --- small keyword maps for the offline fallbacks (clearly NOT an LLM) ---------

_MOOD_WORDS = {
    "dark": "dark", "moody": "dark", "gritty": "dark", "bleak": "dark",
    "cozy": "cozy", "comfort": "cozy", "warm": "cozy", "gentle": "cozy",
    "funny": "lighthearted", "light": "lighthearted", "lighter": "lighthearted",
    "humorous": "lighthearted", "fun": "lighthearted",
    "thoughtful": "thoughtful", "literary": "thoughtful", "emotional": "thoughtful",
    "atmospheric": "atmospheric", "mysterious": "atmospheric",
    "adventurous": "adventurous", "epic": "adventurous",
    "romantic": "romantic", "tender": "romantic",
}

_PACE_WORDS = {
    "fast": "fast", "fast-paced": "fast", "quick": "fast", "page-turner": "fast",
    "thrilling": "fast", "action": "fast",
    "slow": "slow", "slow-burn": "slow", "meandering": "slow", "quiet": "slow",
}

_GENRE_WORDS = {
    "mystery", "thriller", "romance", "fantasy", "sci-fi", "science",
    "horror", "historical", "literary", "adventure", "memoir", "nonfiction",
    "biography", "poetry", "dystopian", "crime", "academia",
}

_RECENCY_WORDS = {
    "recent": "recent", "new": "recent", "modern": "recent", "contemporary": "recent",
    "classic": "classic", "older": "classic", "vintage": "classic", "old": "classic",
}

# words that introduce something to avoid: "no romance", "without horror", ...
_AVOID_RE = re.compile(r"\b(?:no|not|without|avoid|less|skip|fewer)\s+([a-z-]+)", re.I)
_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z-]+")


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

    def facet_terms(self) -> set:
        """Lower-cased positive-signal tokens used by the heuristic scorer."""
        terms = set()
        for value in (self.mood, self.pace, self.recency):
            if value and value != "any":
                terms.add(value.lower())
        for bucket in (self.genres, self.themes):
            for item in bucket:
                terms.update(w.lower() for w in _WORD_RE.findall(str(item)))
        return terms


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


def _intent_fallback(user_turns) -> Intent:
    """Deterministic keyword tagging, weighted to the newest turn."""
    summary = _compose_summary(user_turns)
    # Newest turn carries the strongest signal; scan all turns for tags but let
    # the latest override single-valued facets (mood / pace / recency).
    mood = pace = ""
    recency = "any"
    genres: list = []
    themes: list = []
    avoid: list = []

    # Collect avoid terms first so positive facets can exclude them (e.g. a
    # "no romance" turn must not also tag romance as a wanted genre).
    for turn in user_turns:
        for m in _AVOID_RE.findall(turn.lower()):
            term = m.lower()
            if term not in avoid and term not in {"the", "a", "an", "too"}:
                avoid.append(term)

    for turn in user_turns:
        # strip "no X" / "without X" spans so their object isn't read as a want
        text = _AVOID_RE.sub(" ", turn.lower())
        for word in _WORD_RE.findall(text):
            if word in avoid:
                continue
            if word in _MOOD_WORDS:
                mood = _MOOD_WORDS[word]
            if word in _PACE_WORDS:
                pace = _PACE_WORDS[word]
            if word in _RECENCY_WORDS:
                recency = _RECENCY_WORDS[word]
            if word in _GENRE_WORDS and word not in genres:
                genres.append(word)

    # themes: salient non-stopword tokens from the newest turn not already tagged
    newest = _AVOID_RE.sub(" ", user_turns[-1].lower())
    skip = set(genres) | set(avoid) | {mood, pace, recency} | {
        "book", "books", "read", "reading", "something", "want", "like", "more",
        "with", "and", "the", "for", "that", "make", "them", "some", "give",
    }
    for word in _WORD_RE.findall(newest):
        if len(word) > 3 and word not in skip and word not in _MOOD_WORDS \
                and word not in _PACE_WORDS and word not in _RECENCY_WORDS:
            if word not in themes:
                themes.append(word)
        if len(themes) >= 4:
            break

    return Intent(
        mood=mood,
        genres=genres,
        themes=themes,
        pace=pace or "any",
        avoid=avoid,
        recency=recency,
        summary=summary,
    )


def extract_intent(user_turns, *, model: str = DEFAULT_MODEL,
                   force_fallback: bool = False):
    """Stage A. Returns ``(Intent, used_llm)``. Never raises."""
    turns = [t for t in user_turns if t and t.strip()]
    if not turns:
        return Intent(), False

    if force_fallback or not llm_rerank._have_gemini():
        return _intent_fallback(turns), False

    try:
        data = llm_rerank._structured(_intent_prompt(turns), IntentSchema, model)
    except Exception:
        return _intent_fallback(turns), False

    if not data:
        return _intent_fallback(turns), False

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


# Priority-ordered canned questions for the offline fallback: each entry is
# (still_missing(intent) -> bool, prompt, options).
_CLARIFY_FALLBACKS = [
    (
        lambda i: not i.genres,
        "What kind of story are you in the mood for?",
        ["Mystery or thriller", "Fantasy or sci-fi", "Literary fiction", "Romance"],
    ),
    (
        lambda i: not i.mood,
        "What tone are you after?",
        ["Dark and moody", "Cozy and comforting", "Light and funny", "Atmospheric"],
    ),
    (
        lambda i: i.pace == "any",
        "How fast-paced should it feel?",
        ["Fast page-turner", "Slow burn", "Either is fine"],
    ),
]


def _clarify_fallback(intent: Intent) -> ClarifyingQuestion:
    """Deterministic clarifying question: ask about the most useful missing facet."""
    for missing, prompt, options in _CLARIFY_FALLBACKS:
        if missing(intent):
            return ClarifyingQuestion(
                prompt=prompt,
                options=list(options),
                reason="heuristic gate: request lacked a key facet",
            )
    # Everything common is present but specificity is still low — ask broadly.
    return ClarifyingQuestion(
        prompt="Tell me a bit more about what you're looking for.",
        options=["Surprise me", "Something acclaimed", "A quick read"],
        reason="heuristic gate: request was too general",
    )


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


def build_clarifying_question(intent: Intent, *, model: str = DEFAULT_MODEL,
                              force_fallback: bool = False):
    """Clarify gate. Returns ``(ClarifyingQuestion, used_llm)``. Never raises."""
    if force_fallback or not llm_rerank._have_gemini():
        return _clarify_fallback(intent), False

    try:
        data = llm_rerank._structured(_clarify_prompt(intent), ClarifySchema, model)
    except Exception:
        return _clarify_fallback(intent), False

    prompt = str(data.get("question", "")).strip()
    options = _as_str_list(data.get("options"))
    if not prompt or not options:
        return _clarify_fallback(intent), False
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


def _score_fallback(candidates, intent: Intent):
    """Term-overlap relevance with an avoid penalty (generalizes
    ``llm_rerank._heuristic_rerank`` to a per-candidate score)."""
    terms = intent.facet_terms()
    avoid = {a.lower() for a in intent.avoid}
    scored = []
    for c in candidates:
        text = f"{c.get('title', '')} {c.get('authors', '')}".lower()
        hits = sum(1 for t in terms if t and t in text)
        flags = sorted({f"avoid:{a}" for a in avoid if a and a in text})
        # normalize overlap to 0..1; blend a small CF-score prior so books never
        # all tie at zero on a vague request.
        base = hits / len(terms) if terms else 0.0
        cf = float(c.get("cf_score", 0.0))
        relevance = max(0.0, min(1.0, 0.7 * base + 0.3 * cf - 0.5 * bool(flags)))
        if hits:
            facet = next((t for t in terms if t in text), "")
            reason = f"matches the “{facet}” signal in its title/author" if facet \
                else "matches your stated preference"
        else:
            reason = "carried over as a strong collaborative-filtering pick"
        scored.append(
            ScoredCandidate(
                book_id=c["book_id"],
                relevance=round(relevance, 3),
                reason=reason,
                flags=flags,
            )
        )
    return scored


def score_candidates(candidates, intent: Intent, *, model: str = DEFAULT_MODEL,
                     force_fallback: bool = False):
    """Stage B. Returns ``(list[ScoredCandidate], used_llm)``. Never raises."""
    if not candidates:
        return [], False

    by_id = {c["book_id"]: c for c in candidates}

    if force_fallback or not llm_rerank._have_gemini():
        return _score_fallback(candidates, intent), False

    try:
        parsed = llm_rerank._structured(
            _score_prompt(candidates, intent), SCORE_LIST_SCHEMA, model,
            is_list=True, system=llm_rerank.PERSONA,
        )
    except Exception:
        return _score_fallback(candidates, intent), False

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
        return _score_fallback(candidates, intent), False

    # backfill any candidate the model skipped so Stage C still sees every book
    if len(scored) < len(candidates):
        missing = [c for c in candidates if c["book_id"] not in seen]
        scored.extend(_score_fallback(missing, intent))
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


def _intent_phrase(intent: Intent) -> str:
    """A short, human phrase for the intent (for fallback explanations) — not
    the verbose composed summary."""
    facets = []
    for value in (intent.mood, intent.pace, intent.recency):
        if value and value != "any":
            facets.append(value)
    facets.extend(intent.genres[:2])
    facets.extend(intent.themes[:1])
    phrase = ", ".join(dict.fromkeys(facets))   # de-dupe, preserve order
    if intent.avoid:
        phrase += (", " if phrase else "") + "avoiding " + ", ".join(intent.avoid[:2])
    return phrase


def _coerce_num(value):
    """Best-effort float from a candidate field that may be '?'/None/str."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# Rank-specific framing so each pick reads differently even when two books match
# for the same underlying reason.
_RANK_PHRASES = {
    1: "the collaborative filter's single strongest match for this reader",
    2: "a close second from readers with similar taste",
    3: "a strong third from readers with similar taste",
}


def _heuristic_why(c, scored: ScoredCandidate, intent: Intent, rank: int) -> str:
    """Compose a DISTINCT, book-specific explanation for the offline fallback.

    The live LLM writes these when a key is set; without one we synthesize a
    reason from the book's own metadata — any matched intent terms, its era, its
    average rating — plus its rank, so every pick reads differently instead of
    repeating one generic line.
    """
    text = f"{c.get('title', '')} {c.get('authors', '')}".lower()
    bits = []

    # Strongest signal: grounded in the reader's own history (content-similarity).
    similar_to = c.get("similar_to")
    if similar_to:
        bits.append(f"in the spirit of “{similar_to}” from your favorites")

    matched = [t for t in intent.facet_terms() if t and t in text]
    if matched:
        uniq = list(dict.fromkeys(matched))[:2]
        bits.append("its title signals " + " and ".join(f"“{m}”" for m in uniq))

    year = _coerce_num(c.get("year"))
    if year:
        yr = int(year)
        if intent.recency == "recent" and yr >= 2010:
            bits.append(f"a recent {yr} title for your lean toward newer books")
        elif intent.recency == "classic" and yr < 2000:
            bits.append(f"a {yr} classic matching your taste for older books")

    avg = _coerce_num(c.get("average_rating"))
    if avg:
        bits.append(f"highly rated at {avg:g}/5" if avg >= 4.3
                    else f"rated {avg:g}/5 by readers")

    rank_phrase = _RANK_PHRASES.get(rank, f"ranked #{rank} by readers with similar taste")

    if scored.flags:                               # an avoid-term slipped through
        cleaned = ", ".join(f.replace("avoid:", "") for f in scored.flags)
        return (f"Kept lower because it leans into {cleaned}, which you asked to "
                f"avoid — {rank_phrase}")
    if bits:
        why = bits[0][0].upper() + bits[0][1:]
        if len(bits) > 1:
            why += f", {bits[1]}"
        return f"{why} — {rank_phrase}"
    phrase = _intent_phrase(intent)
    return rank_phrase + (f", fitting your taste for {phrase}" if phrase else "")


def _rerank_fallback(candidates, scored, intent: Intent, top_k: int):
    """Sort by Stage B relevance (avoid-flags last), then CF score, and give each
    pick a distinct, metadata-grounded reason via ``_heuristic_why``."""
    by_id = {c["book_id"]: c for c in candidates}
    ranked = sorted(
        scored,
        key=lambda s: (
            -1 if s.flags else 0,                  # flagged books sink
            s.relevance,
            float(by_id.get(s.book_id, {}).get("cf_score", 0.0)),
        ),
        reverse=True,
    )
    picks = []
    for rank, s in enumerate(ranked[:top_k], start=1):
        c = by_id.get(s.book_id, {})
        picks.append(
            RerankedPick(
                book_id=s.book_id,
                title=c.get("title", ""),
                authors=c.get("authors", ""),
                explanation=_heuristic_why(c, s, intent, rank),
                description=_describe_book(c),
            )
        )
    return picks


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
                       model: str = DEFAULT_MODEL, force_fallback: bool = False):
    """Stage C. Returns ``(list[RerankedPick], used_llm)``. Never raises."""
    if not scored:
        return [], False

    by_id = {c["book_id"]: c for c in candidates}

    if force_fallback or not llm_rerank._have_gemini():
        return _rerank_fallback(candidates, scored, intent, top_k), False

    try:
        parsed = llm_rerank._structured(
            _rerank_prompt(candidates, scored, intent, top_k),
            llm_rerank.RERANK_LIST_SCHEMA, model, is_list=True,
            system=llm_rerank.PERSONA,
        )
    except Exception:
        return _rerank_fallback(candidates, scored, intent, top_k), False

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
        return _rerank_fallback(candidates, scored, intent, top_k), False
    return picks, True


# --- orchestrator --------------------------------------------------------------

def _stage_source(used_a: bool, used_b: bool, used_c: bool, model: str) -> str:
    """Display label mirroring the convention in app.py / llm_rerank."""
    flags = [used_a, used_b, used_c]
    if all(flags):
        return f"Gemini · {model} · 3-stage DAG"
    if not any(flags):
        label = "Heuristic DAG fallback"
        if not os.environ.get("GEMINI_API_KEY"):
            label += " (set GEMINI_API_KEY for live LLM re-ranking)"
        return label
    names = ["Intent", "Scoring", "Re-rank"]
    heuristic = ", ".join(n for n, f in zip(names, flags) if not f)
    return f"Gemini · {model} · partial DAG (heuristic: {heuristic})"


def _clarify_source(used_a: bool, used_gate: bool, model: str) -> str:
    """Display label for a run that paused at the clarify gate."""
    if used_a or used_gate:
        return f"Gemini · {model} · paused for clarification"
    label = "Heuristic gate · paused for clarification"
    if not os.environ.get("GEMINI_API_KEY"):
        label += " (set GEMINI_API_KEY for live LLM)"
    return label


def run_pipeline(user_turns, candidates, top_k: int = 5, *,
                 model: str = DEFAULT_MODEL,
                 force_fallback: bool = False,
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
    picks. Either way a per-stage trace is included. Never raises — each stage
    degrades to its heuristic independently.
    """
    # Stage A
    intent, used_a = extract_intent(
        user_turns, model=model, force_fallback=force_fallback
    )
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
        question, used_gate = build_clarifying_question(
            intent, model=model, force_fallback=force_fallback
        )
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
            used_llm=used_a or used_gate,
            trace=trace,
            source=_clarify_source(used_a, used_gate, model),
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
    scored, used_b = score_candidates(
        candidates, intent, model=model, force_fallback=force_fallback
    )
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
        candidates, scored, intent, top_k=fetch_k,
        model=model, force_fallback=force_fallback,
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

    used_llm = used_a or used_b or used_c
    return PipelineResult(
        picks=picks,
        intent=intent,
        scored=scored,
        used_llm=used_llm,
        trace=trace,
        source=_stage_source(used_a, used_b, used_c, model),
    )
