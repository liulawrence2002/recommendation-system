"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useReducedMotion } from "framer-motion";
import type { ShelfItem } from "@/lib/types";
import { clampIndex } from "@/lib/carousel";
import { formatNumber, formatScore, formatYear } from "@/lib/format";
import CoverFlow from "./CoverFlow";
import BookSpread from "./BookSpread";

export default function Shelf({
  items,
  modelLabel,
  reranked,
  pending,
  onRefine,
}: {
  items: ShelfItem[] | null;
  modelLabel: string;
  reranked: boolean;
  pending: boolean;
  onRefine: () => void;
}) {
  const reduce = useReducedMotion() ?? false;
  const [index, setIndex] = useState(0);
  const [openId, setOpenId] = useState<number | null>(null);
  const firstMount = useRef(true);
  const itemsRef = useRef(items);
  const indexRef = useRef(0);
  itemsRef.current = items;

  const itemsKey = useMemo(
    () => (items ? items.map((i) => i.book_id).join(",") : "none"),
    [items]
  );

  // New / re-ranked shelf: reset to the first cover and bring the shelf into view.
  // Keyed on itemsKey only (which already encodes order + emptiness), so a refine
  // that returns an identical shelf does not yank the viewport or mis-flip.
  useEffect(() => {
    setIndex(0);
    if (firstMount.current) {
      firstMount.current = false;
      return;
    }
    const its = itemsRef.current;
    if (its && its.length) {
      const el = document.getElementById("shelf");
      if (el) {
        const y = el.getBoundingClientRect().top + window.scrollY - 70;
        window.scrollTo({ top: y, behavior: "smooth" });
      }
    }
  }, [itemsKey]);

  // Arrow-key nav + Enter/Space to open (ignored while the modal is open, or when
  // focus is in a control so we don't hijack typing or button activation).
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (openId != null) return;
      const its = itemsRef.current;
      if (!its || !its.length) return;
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      if (e.key === "ArrowRight") {
        e.preventDefault();
        setIndex((i) => clampIndex(i + 1, its.length));
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        setIndex((i) => clampIndex(i - 1, its.length));
      } else if ((e.key === "Enter" || e.key === " ") && tag !== "BUTTON" && tag !== "A") {
        e.preventDefault();
        const ci = clampIndex(indexRef.current, its.length);
        setOpenId(its[ci].book_id);
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [openId]);

  const safeIndex = items && items.length ? clampIndex(index, items.length) : 0;
  indexRef.current = safeIndex;
  const current = items && items.length ? items[safeIndex] : null;
  const openItem = openId != null && items ? items.find((it) => it.book_id === openId) : null;

  return (
    <section id="shelf" className="shelf-section">
      <div className="shelf-head">
        <div className="eyebrow">{reranked ? "Your shelf · re-ranked from your request" : "The shelf"}</div>
        <h2 className="section-title" style={{ fontSize: "clamp(1.5rem, 2.6vw, 2rem)" }}>
          {reranked ? "Re-ranked for what you asked." : "Your candidate shelf."}
        </h2>
      </div>

      {items === null ? (
        <div className="shelf-empty">
          <div className="shelf-empty-icon" aria-hidden="true">📚</div>
          <div className="empty-title">Your shelf is waiting</div>
          <div>
            Pick a few authors or decades above and build a candidate shelf — collaborative
            filtering does the rest.
          </div>
        </div>
      ) : items.length === 0 ? (
        <div className="shelf-empty">
          <div className="empty-title">No candidates matched</div>
          <div>
            Those filters left nothing on the shelf. Relax an author, decade, or the minimum
            ratings, then build again.
          </div>
        </div>
      ) : (
        <>
          <div className="shelf-label">
            <div className="eyebrow" style={{ letterSpacing: "0.16em" }}>
              {reranked ? "re-ranked from your request" : `${modelLabel} candidates`}
            </div>
            <div className="shelf-hint">
              {reduce ? "Scroll the row, or tap a cover to open" : "Drag, use ← → , or tap a spine to open"}
              {pending && <span className="shelf-pending"> · re-ranking…</span>}
            </div>
          </div>

          <CoverFlow
            key={itemsKey}
            items={items}
            index={safeIndex}
            onIndex={setIndex}
            onOpen={(i) => setOpenId(items[i].book_id)}
            reducedMotion={reduce}
          />

          {current && (
            <div className="shelf-caption" key={current.book_id}>
              <div className="shelf-cap-eyebrow">
                <span>
                  {current.match != null
                    ? `${current.match}% match`
                    : `Rank ${String(safeIndex + 1).padStart(2, "0")}`}
                </span>
                <span className="dot">·</span>
                <span>{reranked ? "readers like you" : modelLabel}</span>
              </div>
              <h3 className="shelf-cap-title">{current.title}</h3>
              <div className="shelf-cap-author">by {current.authors}</div>
              <div className="shelf-cap-meta">
                <span>{formatYear(current.book?.year)}</span>
                {current.book && current.book.avgRating > 0 && <span>Avg {current.book.avgRating.toFixed(2)}</span>}
                {current.book && current.book.goodreadsCount > 0 && (
                  <span>{formatNumber(current.book.goodreadsCount)} ratings</span>
                )}
                {current.cfScore != null && <span>CF {formatScore(current.cfScore)}</span>}
              </div>
              <p className="shelf-cap-why">
                {current.why ||
                  `A top ${modelLabel} match for your taste — ranked #${current.rank} on your shelf.`}
              </p>
              <div className="shelf-cap-actions">
                <button className="btn btn-primary" onClick={() => setOpenId(current.book_id)}>
                  Open this book ↗
                </button>
                <button className="btn btn-secondary" onClick={onRefine}>
                  Refine in chat
                </button>
              </div>
            </div>
          )}

          <div className="shelf-dots">
            {items.map((it, i) => (
              <button
                key={it.book_id}
                className={`shelf-dot${i === safeIndex ? " is-active" : ""}`}
                aria-label={`Go to ${it.title}`}
                aria-current={i === safeIndex}
                onClick={() => setIndex(i)}
              />
            ))}
          </div>
        </>
      )}

      {openItem && (
        <BookSpread
          item={openItem}
          modelLabel={modelLabel}
          onClose={() => setOpenId(null)}
          onRefine={onRefine}
        />
      )}
    </section>
  );
}
