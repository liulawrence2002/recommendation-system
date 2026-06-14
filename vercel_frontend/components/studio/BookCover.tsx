"use client";

import { memo, useMemo, useState } from "react";
import type { Book } from "@/lib/types";
import { clothFor, coverCandidates } from "@/lib/carousel";

/* A book-cover face: a typographic cloth cover (always rendered) with the real
   Goodreads image layered on top. Requests the LARGE Goodreads variant for a
   crisp cover and walks down to medium/small/cloth on a 404 — so covers are sharp
   on the ~210px cards, never blurry. Memoized so it doesn't re-render on drag. */
function BookCoverImpl({ book, big = false }: { book?: Book; big?: boolean }) {
  const candidates = useMemo(
    () => coverCandidates(book?.cover, book?.image),
    [book?.cover, book?.image]
  );
  const [idx, setIdx] = useState(0);
  const cloth = clothFor(book?.id ?? 0);
  const src = candidates[idx] ?? "";
  const title = book?.title || "Untitled";
  const authors = book?.authors || "Unknown";
  const kicker = book?.year && book.year > 0 ? String(book.year) : "BookRec";

  return (
    <div className={`cover-face${big ? " cover-face--big" : ""}`} style={{ background: cloth }}>
      <div
        className="cover-cloth"
        style={{ background: `linear-gradient(150deg, ${cloth} 0%, rgba(0,0,0,0.42) 130%)` }}
      >
        <div className="cover-kicker">{kicker}</div>
        <div className="cover-title">{title}</div>
        <div className="cover-author">{authors}</div>
      </div>
      {src && (
        <img
          key={src}
          className="cover-img"
          src={src}
          alt=""
          loading="lazy"
          decoding="async"
          referrerPolicy="no-referrer"
          onError={() => setIdx((i) => i + 1)}
        />
      )}
      <div className="cover-gloss" />
      <div className="cover-spine" />
    </div>
  );
}

export default memo(BookCoverImpl);
