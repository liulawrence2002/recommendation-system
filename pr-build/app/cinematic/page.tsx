// app/cinematic/page.tsx
//
// Mounts the self-contained cinematic BookRec experience that lives at
// public/cinematic/index.html. The build is a single offline-capable HTML
// file, so we render it full-bleed in an iframe — no extra dependencies,
// no hydration concerns, and it can't interfere with the rest of the app.
//
// To make this the site's landing page instead of a /cinematic route,
// move this file's <CinematicPage/> body into app/page.tsx (or have
// app/page.tsx re-export it).

import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "BookRec — Find your next book",
  description:
    "A cinematic, conversational book-recommendation experience: filter by author and decade, let collaborative filtering build your shelf, then refine it in conversation.",
};

export default function CinematicPage() {
  return (
    <iframe
      src="/cinematic/index.html"
      title="BookRec — Cinematic"
      style={{
        position: "fixed",
        inset: 0,
        width: "100%",
        height: "100%",
        border: "none",
      }}
      allow="fullscreen"
    />
  );
}
