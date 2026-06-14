# Pull request

**Suggested branch:** `feat/cinematic-landing`

**Title:** Cinematic landing experience — filter → CF shelf → conversational refine

---

## Summary

Reworks the BookRec front end into a cinematic, single-experience landing page,
while preserving the original product flow (filter the candidate pool with
collaborative filtering, then refine in conversation).

- **Cold open**: an empty reading-room-at-noon title sequence (letterboxed,
  light beams, dust motes, an empty chair, steaming coffee) that dissolves into
  the page on *Take your seat*.
- **Filter panel** (carried over from the studio): a **searchable Authors
  dropdown** (type-to-filter, removable chips), **Decades** chips, and
  **Advanced options** (model select, candidate count, minimum average rating).
- **Cover-flow shelf**: the collaborative-filtering candidates as a 3D,
  swipeable carousel — drag / arrow keys / tap. Advancing plays a page-flip;
  tapping a book opens a 3D two-page spread (synopsis + "why this pick").
- **Conversational refine (RAG)**: re-ranks **only the candidates on the shelf**
  (never invents a title), with the reasoning shown in a thread.

## How it ships

- `public/cinematic/index.html` — the experience as one self-contained,
  offline-capable file (no new npm deps).
- `app/cinematic/page.tsx` — app-router route that mounts it full-bleed at
  **`/cinematic`**.

## How to test

1. `npm run dev`
2. Visit `/cinematic`.
3. Click *Take your seat* → pick an author (type to search) and/or a decade →
   *Build my shelf* → swipe the carousel, open a book → use the composer to
   refine (e.g. "a cozy mystery, nothing gory").

## Notes / follow-ups

- CF scoring and the RAG re-rank currently run client-side on a ten-book sample
  so the page works with no backend. To wire the real pipeline, swap the local
  `generate()` / `send()` logic for the existing candidate + `/api/chat`
  endpoints (see README → "Wiring it to the real recommender").
- Book covers load from OpenLibrary by ISBN with a typographic fallback; swap in
  Goodreads URLs via `coverUrl(b)`.
- Built/verified in Chromium; honors `prefers-reduced-motion`.

## Suggested commit message

```
feat(web): cinematic landing — filter → CF shelf → conversational refine

Add a self-contained cinematic landing experience (public/cinematic/index.html)
and an app-router route (app/cinematic/page.tsx) that mounts it at /cinematic.

- Cold-open title sequence (empty reading room at noon)
- Searchable author dropdown + decade/model filter panel
- 3D cover-flow carousel of the CF candidate shelf with page-flip + book-open
- Conversational RAG refine that re-ranks only the on-shelf candidates

Client-side sample scoring for now; README documents wiring to the real
candidate + /api/chat pipeline. No new dependencies.
```
