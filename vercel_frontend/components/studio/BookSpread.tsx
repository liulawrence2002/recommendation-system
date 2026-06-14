"use client";

import { useEffect, useId, useRef, useState } from "react";
import { createPortal } from "react-dom";
import type React from "react";
import type { ShelfItem } from "@/lib/types";
import { formatNumber, formatYear } from "@/lib/format";
import BookCover from "./BookCover";

/* The 3D two-page "book spread" modal. Left page = why this pick, right page =
   synopsis + meta, with a front cover that flips open. Focus-trapped, Escape /
   backdrop close, body-scroll locked, focus restored to the opener. */
export default function BookSpread({
  item,
  modelLabel,
  onClose,
  onRefine,
}: {
  item: ShelfItem;
  modelLabel: string;
  onClose: () => void;
  onRefine: () => void;
}) {
  const [closing, setClosing] = useState(false);
  const dialogRef = useRef<HTMLDivElement>(null);
  const closeRef = useRef<HTMLButtonElement>(null);
  const closeTO = useRef<ReturnType<typeof setTimeout> | null>(null);
  const titleId = useId();

  const book = item.book;
  const avg = book && book.avgRating > 0 ? book.avgRating.toFixed(2) : null;
  const why =
    item.why || `Ranked onto your shelf by collaborative filtering (${modelLabel}).`;
  const synopsis =
    item.synopsis ||
    `${item.title} by ${item.authors}${
      book && book.year > 0 ? `, first published in ${book.year}` : ""
    }${avg ? `, a reader-rated title averaging ${avg}/5` : ""}.`;

  function beginClose() {
    if (closing) return;
    setClosing(true);
    closeTO.current = setTimeout(onClose, 280);
  }

  // Mount: lock scroll, capture + move focus. Unmount: restore.
  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null;
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    closeRef.current?.focus();
    return () => {
      document.body.style.overflow = prevOverflow;
      if (closeTO.current) clearTimeout(closeTO.current);
      opener?.focus?.();
    };
  }, []);

  function onKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Escape") {
      e.stopPropagation();
      beginClose();
      return;
    }
    if (e.key !== "Tab") return;
    const focusables = dialogRef.current?.querySelectorAll<HTMLElement>(
      'button, a[href], [tabindex]:not([tabindex="-1"])'
    );
    if (!focusables || focusables.length === 0) return;
    const first = focusables[0];
    const last = focusables[focusables.length - 1];
    if (e.shiftKey && document.activeElement === first) {
      e.preventDefault();
      last.focus();
    } else if (!e.shiftKey && document.activeElement === last) {
      e.preventDefault();
      first.focus();
    }
  }

  return createPortal(
    <div
      className={`book-spread-overlay${closing ? " is-closing" : ""}`}
      onClick={beginClose}
      onKeyDown={onKeyDown}
    >
      <div
        ref={dialogRef}
        className={`book-spread${closing ? " is-closing" : ""}`}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="spread-page spread-left">
          <div className="spread-eyebrow">Why this pick</div>
          <div className="spread-why">{why}</div>
          <div className="spread-meta">
            {avg && <span>★ {avg}</span>}
            {book && book.goodreadsCount > 0 && <span>{formatNumber(book.goodreadsCount)} ratings</span>}
            {book && book.year > 0 && <span>{formatYear(book.year)}</span>}
            {item.match != null && <span>{item.match}% match</span>}
          </div>
        </div>

        <div className="spread-page spread-right">
          <div className="spread-eyebrow">{item.match != null ? "On your shelf" : modelLabel}</div>
          <h3 id={titleId} className="spread-title">
            {item.title}
          </h3>
          <div className="spread-author">by {item.authors}</div>
          <p className="spread-synopsis">{synopsis}</p>
          <div className="spread-spacer" />
          <div className="spread-actions">
            <button
              className="btn btn-primary"
              onClick={() => {
                onRefine();
                beginClose();
              }}
            >
              Refine in chat →
            </button>
            {book?.url && (
              <a className="btn btn-secondary" href={book.url} target="_blank" rel="noopener noreferrer">
                Goodreads ↗
              </a>
            )}
            <button ref={closeRef} className="btn btn-secondary" onClick={beginClose}>
              Close
            </button>
          </div>
        </div>

        <div className="spread-cover" aria-hidden="true">
          <BookCover book={book} big />
        </div>
      </div>
    </div>,
    document.body
  );
}
