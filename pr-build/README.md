# BookRec — Cinematic landing experience

A self-contained, cinematic reworking of the BookRec front end. It keeps the
project's warm library palette (cream `#f4ecdd`, leather/amber `#a35421` ·
`#c9802f`, Fraunces + Inter) and the original product flow, presented as a
film-style experience:

1. **Cold open** — an empty reading room at noon (light beams, dust motes, an
   empty chair, steaming coffee) with a fading title card. Click *Take your
   seat* to enter.
2. **Filter → collaborative filtering** — the original Filter panel: a
   **searchable Authors dropdown** (type to filter), **Decades** chips, and
   **Advanced options** (model: UBCF / IBCF / baseline / SVD / popularity,
   candidate count, minimum average rating). "Build my shelf" runs the chosen
   model over the filtered pool.
3. **The shelf** — a 3D cover-flow carousel of the candidate books. Drag,
   arrow keys, or tap a spine. Advancing plays a page-flip; tapping a book
   opens a 3D two-page spread (synopsis + "why this pick").
4. **Refine (RAG)** — a conversational composer re-ranks **only the candidates
   already on the shelf** (never invents a title), showing the reasoning
   (intent chips, "readers like you" note) in a thread.

## What's in this package

```
public/cinematic/index.html      ← the experience, fully self-contained (offline-capable)
app/cinematic/page.tsx           ← Next.js (app router) route that mounts it at /cinematic
source/BookRec Cinematic.dc.html ← editable source (Design Component markup + logic)
source/support.js                ← runtime the source needs to render in isolation
PR.md                            ← suggested branch name, PR title & description, commit message
```

## How to integrate

1. Copy `public/cinematic/` and `app/cinematic/` into your `vercel_frontend`
   project at the same paths.
2. Run the app (`npm run dev`) and visit **`/cinematic`**.
   - The route renders `public/cinematic/index.html` full-bleed in an iframe.
   - The static file and the `/cinematic` route do not conflict (different URLs).
3. To make it the **home page**, point `app/page.tsx` at the same iframe (or
   move the body of `app/cinematic/page.tsx` into it).

No new npm dependencies are required.

## Editing the experience

`public/cinematic/index.html` is a **compiled** bundle — don't hand-edit it.
Edit `source/BookRec Cinematic.dc.html` instead (it's a Design Component:
inline-styled markup plus a `Component` logic class), then recompile it into a
single self-contained file and drop the result at `public/cinematic/index.html`.

### The featured books & covers

The ten featured titles, their copy ("why this pick"), tags, and ISBNs live in
the `books` array inside `source/BookRec Cinematic.dc.html`. Covers load from
`https://covers.openlibrary.org/b/isbn/<isbn>-L.jpg`, with a typographic
cloth-cover fallback if an image 404s. Swap in your own Goodreads cover URLs by
editing `coverUrl(b)` or adding a `cover` field per book.

### Wiring it to the real recommender

Today the collaborative-filtering scores and the RAG re-rank run client-side on
the ten-book sample (model weights + intent keyword matching) so the page works
with zero backend. To use the real pipeline, replace:

- **`generate()`** — call your candidate endpoint (filtered by author/decade/
  model) instead of the local `modelBase()` ranking.
- **`send()`** — POST the user's turn + current candidates to `/api/chat`
  (as `StudioApp.submit` does) and apply the returned ordering, instead of the
  local intent scoring.

## Notes

- Built and verified in Chromium. Uses modern CSS (`animation-timeline: view()`
  for scroll reveals, 3D transforms); entrance animations pause while the tab is
  backgrounded and play on focus.
- Honors `prefers-reduced-motion`.
- Covers and Google Fonts are fetched at runtime (needs network); everything
  else is inlined.
