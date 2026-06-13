# BookRec — cinematic frontend

A deploy-ready, cinematic rebuild of the Streamlit **Goodreads book recommender**
as a Next.js app. Same framework, new skin:

- **Landing** — a warm, literary canvas with fluid, flowing motion (Whisperflow /
  Oura inspired): drifting gradient mesh, a riffling 3D book, scroll-reveal storytelling.
- **Studio** — the full recommender: filter the catalog, generate collaborative-filtering
  candidates, inspect the EDA + model audit, then refine in a **Claude/OpenAI-style chat**
  powered by a transparent **3-stage RAG/DAG** re-ranker.

It reproduces the original `bookrec/app.py` pipeline 1:1 — and runs on Vercel with
**zero external services required**.

---

## How it works

The heavy ML (scikit-surprise UBCF/IBCF/SVD, live training over 164k ratings) doesn't
belong in a serverless function, so it runs **once, offline**, and bakes the results into
small JSON files the browser reads:

```
scripts/export_data.py   →  public/data/books.json   (catalog: ~9,964 books)
                            public/data/scores.json  (per-book CF score per model,
                                                       for the most-active reader)
                            public/data/stats.json   (EDA, model audit, filter options)
```

With those scores baked in, the browser reproduces `recommend.recommend_top_n` exactly
(popularity floor → exclude seen → author/decade filter → sort by the chosen model) —
interactively, with no Python at request time.

The conversational re-ranker (`lib/dag/`) is a faithful TypeScript port of
`bookrec/src/rag_pipeline.py`:

```
your message → [A] extract intent → [clarify gate?] → [B] score candidates → [C] re-rank
```

Every stage has a **deterministic heuristic fallback** (so it always answers) and an
**optional live Gemini path** (when `GEMINI_API_KEY` is set). The LLM only ever re-ranks
the real CF candidates — it can never invent a title.

---

## Run it locally

```bash
cd vercel_frontend
npm install
npm run dev          # http://localhost:3000
```

The JSON data files are committed under `public/data`, so it runs out of the box.

### Regenerate the data (optional)

Only needed if the underlying CSVs change. Requires Python with
`pandas`, `scipy`, `scikit-learn` (the repo's data lives in `../bookrec/data`):

```bash
npm run export-data      # == python scripts/export_data.py
```

### Enable the live LLM layer (optional)

```bash
cp .env.example .env.local
# then set GEMINI_API_KEY=...   (free key: https://aistudio.google.com/app/apikey)
```

Without a key the app uses the heuristic DAG — fully functional, just rule-based.

---

## Deploy to Vercel

1. Push this repo to GitHub.
2. In Vercel, **New Project → import the repo**.
3. **Set the Root Directory to `vercel_frontend`** (this app is a subfolder of the
   recommendation-system repo). Framework preset: **Next.js** (auto-detected).
4. *(Optional)* Add an Environment Variable `GEMINI_API_KEY` to enable live re-ranking.
5. Deploy. No build-time Python is required — the `public/data/*.json` files are committed.

That's it: the landing is statically generated, the studio is interactive client-side, and
`/api/chat` runs as a serverless function.

---

## Project layout

```
app/
  page.tsx              landing (static)
  studio/page.tsx       the recommender app (dynamic; reads GEMINI_API_KEY for the nav badge)
  api/chat/route.ts     runs the RAG/DAG per turn (Gemini or heuristic)
components/             landing sections + studio UI (filters, EDA, candidates, chat)
lib/
  recommend.ts          port of recommend_top_n + filtering
  dag/                  port of rag_pipeline.py (intent → clarify → score → rerank)
  gemini.ts             server-only Gemini REST call
  clientData.ts         fetches /data/*.json in the browser
scripts/export_data.py  offline data bake (reuses ../bookrec/src/data_loader)
public/data/*.json      generated catalog + scores + stats
```
