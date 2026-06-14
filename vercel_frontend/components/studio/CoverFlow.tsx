"use client";

import { useEffect, useRef, useState } from "react";
import type React from "react";
import type { ShelfItem } from "@/lib/types";
import { CARD_W, CARD_H, SPACING, cardTransform, clampIndex } from "@/lib/carousel";
import BookCover from "./BookCover";

function ariaName(it: ShelfItem): string {
  return `${it.title} by ${it.authors}, rank ${it.rank}${
    it.match != null ? `, ${it.match}% match` : ""
  }`;
}

export default function CoverFlow({
  items,
  index,
  onIndex,
  onOpen,
  reducedMotion,
}: {
  items: ShelfItem[];
  index: number;
  onIndex: (i: number) => void;
  onOpen: (i: number) => void;
  reducedMotion: boolean;
}) {
  const [dragging, setDragging] = useState(false);
  const [dragDX, setDragDX] = useState(0);
  const startX = useRef(0);
  const movedRef = useRef(false); // survives until the synthetic click runs
  const [flip, setFlip] = useState({ dir: 0, key: 0 });
  const flipTO = useRef<ReturnType<typeof setTimeout> | null>(null);
  const prevIndex = useRef(index);

  // Play a directional page-flip whenever the centred card changes.
  useEffect(() => {
    if (reducedMotion) {
      prevIndex.current = index;
      return;
    }
    const dir = Math.sign(index - prevIndex.current);
    prevIndex.current = index;
    if (dir !== 0) {
      setFlip((f) => ({ dir, key: f.key + 1 }));
      if (flipTO.current) clearTimeout(flipTO.current);
      flipTO.current = setTimeout(() => setFlip((f) => ({ ...f, dir: 0 })), 600);
    }
  }, [index, reducedMotion]);

  // A freshly built / re-ranked shelf remounts this component (keyed on itemsKey),
  // so a forward flip on mount reads as "the shelf turned to a new page".
  useEffect(() => {
    if (reducedMotion || items.length < 2) return;
    setFlip((f) => ({ dir: 1, key: f.key + 1 }));
    const t = setTimeout(() => setFlip((f) => ({ ...f, dir: 0 })), 600);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => () => { if (flipTO.current) clearTimeout(flipTO.current); }, []);

  const pos = dragging ? index - dragDX / SPACING : index;

  function onDown(e: React.PointerEvent) {
    startX.current = e.clientX;
    movedRef.current = false;
    setDragging(true);
    setDragDX(0);
    // Capture so move/up keep targeting the track even if the pointer leaves it.
    try {
      (e.currentTarget as Element).setPointerCapture(e.pointerId);
    } catch {
      /* not supported — fine */
    }
  }
  function onMove(e: React.PointerEvent) {
    if (!dragging) return;
    const dx = e.clientX - startX.current;
    if (Math.abs(dx) > 6) movedRef.current = true;
    setDragDX(dx);
  }
  function onUp() {
    if (!dragging) return;
    const ni = clampIndex(Math.round(index - dragDX / SPACING), items.length);
    setDragging(false);
    setDragDX(0);
    if (ni !== index) onIndex(ni);
  }
  function onCancel() {
    setDragging(false);
    setDragDX(0);
  }

  // Reduced motion → a plain horizontal scroll-snap row (no perspective/JS transforms).
  if (reducedMotion) {
    return (
      <div className="shelf-row" role="listbox" aria-label="Your shelf">
        {items.map((it, i) => (
          <button
            key={it.book_id}
            className={`shelf-row-item${i === index ? " is-active" : ""}`}
            role="option"
            aria-selected={i === index}
            aria-label={ariaName(it)}
            onClick={() => {
              onIndex(i);
              onOpen(i);
            }}
          >
            <span className="shelf-row-cover">
              <BookCover book={it.book} />
            </span>
            {it.match != null && <span className="shelf-row-badge">{it.match}%</span>}
          </button>
        ))}
      </div>
    );
  }

  return (
    <div className="coverflow-stage">
      <div className="coverflow-pit" aria-hidden="true" />
      <div className="coverflow-rim" aria-hidden="true" />
      <div
        className="coverflow-track"
        role="listbox"
        aria-label="Your shelf"
        onPointerDown={onDown}
        onPointerMove={onMove}
        onPointerUp={onUp}
        onPointerCancel={onCancel}
      >
        {items.map((it, i) => {
          const t = cardTransform(i - pos);
          const isCenter = Math.round(pos) === i;
          return (
            <div
              key={it.book_id}
              className="coverflow-card"
              role="option"
              aria-selected={isCenter}
              aria-hidden={t.pointer ? undefined : true}
              aria-label={ariaName(it)}
              style={{
                width: CARD_W,
                height: CARD_H,
                marginLeft: -CARD_W / 2,
                marginTop: -CARD_H / 2,
                transform: `translateX(${t.x}px) translateZ(${t.translateZ}px) rotateY(${t.rotateY}deg) scale(${t.scale})`,
                filter: `blur(${t.blur}px) brightness(${t.brightness})`,
                opacity: t.opacity,
                zIndex: t.zIndex,
                transition: dragging
                  ? "none"
                  : "transform .6s cubic-bezier(.2,.8,.2,1), filter .6s, opacity .6s",
                pointerEvents: t.pointer ? "auto" : "none",
              }}
              onClick={(e) => {
                e.stopPropagation();
                if (movedRef.current) {
                  movedRef.current = false; // it was a drag, not a tap
                  return;
                }
                if (isCenter) onOpen(i);
                else onIndex(i);
              }}
            >
              {isCenter && <div className="coverflow-glow" aria-hidden="true" />}
              <BookCover book={it.book} big={isCenter} />
              {it.match != null && (
                <div className="coverflow-match" style={{ opacity: isCenter ? 1 : 0 }}>
                  {it.match}% match
                </div>
              )}
              {isCenter && (
                <div className="coverflow-reflection" aria-hidden="true">
                  <BookCover book={it.book} />
                </div>
              )}
            </div>
          );
        })}
        {flip.dir !== 0 && (
          <div
            key={`flip${flip.key}`}
            className={`coverflow-flip ${flip.dir > 0 ? "flip-fwd" : "flip-back"}`}
            style={{
              width: CARD_W,
              height: CARD_H,
              marginLeft: -CARD_W / 2,
              marginTop: -CARD_H / 2,
              transformOrigin: flip.dir > 0 ? "left center" : "right center",
            }}
            aria-hidden="true"
          >
            <div className={`coverflow-flip-inner ${flip.dir > 0 ? "fwd" : "back"}`} />
          </div>
        )}
      </div>
      <button
        className="coverflow-arrow coverflow-arrow--l"
        aria-label="Previous book"
        onClick={() => onIndex(clampIndex(index - 1, items.length))}
      >
        ‹
      </button>
      <button
        className="coverflow-arrow coverflow-arrow--r"
        aria-label="Next book"
        onClick={() => onIndex(clampIndex(index + 1, items.length))}
      >
        ›
      </button>
    </div>
  );
}
