# LLM re-ranking layer (grounded personalization)

The assignment requires an LLM that **re-ranks** collaborative-filtering Top-N candidates — it must **not invent books**. Implementation: [`src/llm_rerank.py`](../src/llm_rerank.py).

## Flow

1. `recommend.recommend_top_n()` produces CF candidates (`book_id`, title, authors, score).
2. `candidates_from_recs()` attaches metadata (`year`, `average_rating`, `cf_score`).
3. `rerank(candidates, preference)` returns ordered picks + one-sentence explanations.

## Grounding rules (anti-hallucination)

| Rule | How it is enforced |
|------|-------------------|
| Only existing catalog items | Prompt lists allowed `book_id`s; parser accepts only IDs in the candidate set |
| No free-form book titles from the model | Output schema is JSON `[{"book_id": int, "explanation": str}]` |
| Fallback if API fails | Heuristic re-ranker uses the same candidate list |

## Prompt strategy (`build_prompt`)

- **Role**: re-ranker, not generator — "RE-RANK these candidates."
- **Context**: numbered lines `[book_id] "title" by author (year), avg_rating=..., cf_score=...`
- **Constraint**: explicit list of valid IDs; "Do NOT invent books."
- **Output**: strict JSON array, `top_k` items, preference-tied explanations.

**Provider / model (cite in slides):** Google Gemini, `gemini-2.5-flash-lite` (override via `rerank(..., model=...)`).

## Environment setup

```bash
export GEMINI_API_KEY=your_key_here   # Windows: set GEMINI_API_KEY=...
pip install google-generativeai
cd bookrec
streamlit run app.py
```

Without a key or SDK, `rerank()` uses the **heuristic fallback** (keyword overlap on title/author + CF score tie-break). The app labels which path ran.

## Why not a vector store (Chroma / RAG retrieval)?

At ~10k books, candidates come from CF Top-N, not ANN search over embeddings. A vector DB would add ops complexity without improving grounded rerank quality for this rubric. Content vectors stay **in memory** in `content_model.py` (optional hybrid extra).
