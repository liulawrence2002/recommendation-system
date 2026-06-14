"use client";

import {
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { createPortal } from "react-dom";
import { useReducedMotion } from "framer-motion";

/* The cinematic entry: a full-screen video backdrop with "Ready for your next
   book?" — click anywhere (or the button / Enter) to dismiss it into the studio.
   Shown on every load (it's the loading screen). Portaled to <body> so it sits
   above the fixed nav; honors prefers-reduced-motion (no autoplaying video). */
export default function IntroGate({ children }: { children: ReactNode }) {
  const reduce = useReducedMotion() ?? false;
  const [show, setShow] = useState(true);
  const [leaving, setLeaving] = useState(false);
  const [mounted, setMounted] = useState(false);
  const leaveTO = useRef<ReturnType<typeof setTimeout> | null>(null);
  const btnRef = useRef<HTMLButtonElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    setMounted(true);
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = "";
      if (leaveTO.current) clearTimeout(leaveTO.current);
    };
  }, []);

  // Force muted (React's `muted` attribute can be unreliable) and kick off
  // autoplay; if the browser still blocks it, the scrim + title carry the screen.
  useEffect(() => {
    const v = videoRef.current;
    if (v) {
      v.muted = true;
      v.play?.().catch(() => {});
    }
  }, [mounted]);

  function enter() {
    document.body.style.overflow = "";
    try {
      window.scrollTo({ top: 0 });
    } catch {
      /* ignore */
    }
    if (reduce) {
      setShow(false);
      return;
    }
    setLeaving(true);
    leaveTO.current = setTimeout(() => setShow(false), 900);
  }

  // Move focus into the gate; Escape / Enter dismiss it.
  useEffect(() => {
    if (!show || !mounted) return;
    btnRef.current?.focus();
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape" || e.key === "Enter") {
        e.preventDefault();
        enter();
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [show, mounted]);

  const overlay = (
    <div
      className={`intro intro--video${leaving ? " intro--leaving" : ""}`}
      role="dialog"
      aria-modal="true"
      aria-label="Welcome to BookRec"
      onClick={enter}
    >
      {!reduce && (
        <video
          ref={videoRef}
          className="intro-video"
          src="/intro.mp4"
          autoPlay
          muted
          loop
          playsInline
          preload="auto"
          aria-hidden="true"
        />
      )}
      <div className="intro-scrim" />
      <div className="intro-title">
        <div className="intro-kicker">BookRec · a reading room</div>
        <h1 className="intro-h1">
          Ready for <span className="intro-w2">your next book?</span>
        </h1>
        <div className="intro-cta">
          <button
            ref={btnRef}
            className="intro-begin"
            onClick={(e) => {
              e.stopPropagation();
              enter();
            }}
          >
            Click anywhere to begin →
          </button>
        </div>
      </div>
    </div>
  );

  return (
    <>
      {show && mounted && createPortal(overlay, document.body)}
      {children}
    </>
  );
}
