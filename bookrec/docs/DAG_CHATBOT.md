# The DAG Chatbot — Conversational LLM Re-Ranking

How BookRec's chat turns a static collaborative-filtering Top-N into a
personalized, explained shortlist. The logic lives in
[`src/rag_pipeline.py`](../src/rag_pipeline.py); the chat UI is in
[`app.py`](../app.py).

---

## What it is

A **sequential DAG** (directed acyclic graph) of dependent LLM stages. Where the
notebook's `llm_rerank.rerank()` does the whole personalization in **one** LLM
call, the chatbot chains **three dependent stages plus a gate**, so each prompt is
grounded in the structured output of the previous one:

```
                 user chat turns (base request + refinements)
                              │
                              ▼
                   ┌──────────────────────┐
        [A]        │   Extract intent     │  → Intent {mood, genres, themes,
                   │  (structured JSON)   │      pace, avoid, recency, summary}
                   └──────────┬───────────┘
                              │ Intent
                              ▼
                   ┌──────────────────────┐   too vague?
       [gate]      │   Clarify gate       │ ───────────────►  ask ONE question,
                   │  (specificity check) │                   pause, wait for reply
                   └──────────┬───────────┘
                              │ enough signal
                              ▼
                   ┌──────────────────────┐
        [B]        │  Score candidates    │  → ScoredCandidate[] {relevance 0-1,
                   │  (grounded in Intent)│      reason, avoid-flags} per book
                   └──────────┬───────────┘
                              │ scores
                              ▼
                   ┌──────────────────────┐
        [C]        │  Re-rank + explain   │  → ordered RerankedPick[]
                   │  (grounded in scores)│      {description, explanation}
                   └──────────┬───────────┘
                              │
                              ▼
                   diversity trim → final Top-N shown in chat
```

Each stage exchanges **structured objects** (dataclasses), not raw model text, so
stages compose cleanly and can each fall back to a heuristic independently.

---

## The stages

### A — Extract intent
Reads the **whole conversation** (the first message is the base request; later
messages are refinements, newest wins on conflicts) and distills it into a typed
`Intent`: `mood`, `genres`, `themes`, `pace`, `avoid`, `recency`, and a one-line
`summary`. Uses Gemini **structured output** (a Pydantic schema; `pace`/`recency`
are constrained **enums**) so the result is always valid and typed.

### Gate — Clarify when too vague
Counts the distinct signals in the `Intent`. If the request is too thin (below a
threshold) **and** we haven't already asked too many times, the DAG **pauses** and
returns one short clarifying question with tap-to-answer options instead of
guessing. After at most two rounds — or if the user clicks *"Just recommend
something"* — it proceeds anyway.

### B — Score candidates
Scores **how well each CF candidate matches the intent** (relevance 0–1 + a
one-clause reason + flags for any `avoid` terms it triggers). The model is told to
score **every** candidate and never drop or invent one; any candidate the model
skips is back-filled by the heuristic so Stage C always sees the full set.

### C — Re-rank + explain
Uses the upstream scores and the intent to **select and order** the final picks,
writing for each a neutral one-sentence **description** (what the book is) and a
one-sentence **explanation** (why it earns this rank). A **diversity trim** then
caps the list at ≤2 books per author and one per series.

---

## Design principles

- **Grounded, never invented.** The LLM only ever **re-ranks the CF candidates** —
  Stages B and C re-validate every `book_id` against the candidate set, so the
  model can reorder and explain but cannot add books outside the shortlist.
- **Content-similarity grounding.** Each candidate is annotated with the reader's
  own highly-rated book it most resembles (via the TF-IDF `ContentModel`), so
  explanations can tie a pick to something the reader already loved ("in the
  spirit of *Mistborn*"), grounded in real data rather than the model's memory.
- **Structured output (Week-4 technique).** Every stage forces Gemini to return
  JSON matching a Pydantic schema (`response_schema`), so we read typed objects
  via `response.parsed` instead of regex-parsing free text.
- **Persona / system instruction.** The scoring and re-rank stages run with a
  librarian persona that requires each explanation to be **specific to that book**
  and **never repeated** across picks.
- **Graceful fallback everywhere.** Every stage has a deterministic, transparent
  heuristic. With no API key (or on any error/quota limit) the **entire DAG still
  runs offline** — stages fall back independently, and the UI labels which ran on
  the live LLM vs. "smart rules."
- **Caching.** Identical stage prompts are cached in-process, so repeated turns
  return instantly and don't re-bill the API (mitigates cost / latency / rate
  limits).
- **Transparent trace.** Every run records a per-stage `StageTrace`, surfaced in
  the chat under *"How we chose these"* — intent chips, per-book scores, and the
  final ordering, each badged **AI** or **Smart rules**.

---

## Inputs & outputs

**Input:** the ordered list of user chat turns + the CF Top-N candidates (as
dicts with `book_id`, `title`, `authors`, `year`, `average_rating`, `cf_score`,
and optional `similar_to`), plus `top_k`.

**Output:** a `PipelineResult` with either
- `question` set (the DAG paused to clarify), **or**
- `picks` — the ordered, explained, diversity-trimmed shortlist —

always with the `intent`, the per-stage `trace`, a `used_llm` flag, and a `source`
label (e.g. `Gemini · gemini-2.5-flash-lite · 3-stage DAG` or `Heuristic DAG
fallback`).

---

## Why a DAG instead of one call?

| One-shot call (`llm_rerank.rerank`) | Sequential DAG (`rag_pipeline.run_pipeline`) |
|---|---|
| One prompt does everything | Each stage has one clear job, grounded in the last |
| No conversation handling | Folds multiple turns; newest refinement wins |
| Ranks even vague requests | Asks a clarifying question first when needed |
| One reason per pick | Intent → per-book scores → ordered, contrasted reasons |
| All-or-nothing if the call fails | Stages fall back independently; always returns |

The one-shot version is kept for the notebook; the app uses the DAG.

---

## Assignment mapping

This is the **AI re-ranking layer** required by the project: it takes the Top-N
from the best collaborative-filtering model and uses an LLM to **re-rank and
personalize** them to the user's stated preference, returning a short explanation
per pick — using book metadata as context and **never** generating new
recommendations outside the CF candidate set. Provider/model:
**Google `gemini-2.5-flash-lite`**; the API key is read from the environment and
never committed.

**Key files:** [`src/rag_pipeline.py`](../src/rag_pipeline.py) (the DAG),
[`src/llm_rerank.py`](../src/llm_rerank.py) (Gemini structured-output helper,
persona, schemas, one-shot re-ranker), [`app.py`](../app.py) (chat UI + trace).
