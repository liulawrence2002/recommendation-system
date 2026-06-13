"use client";

import { useEffect } from "react";

/** Publishes scroll-derived CSS offsets on the root element. The fixed background
    layers (.par-slow / .par-fast) translate by those offsets for cinematic depth.
    rAF-throttled; disabled under
    prefers-reduced-motion. Renders nothing. */
export default function ParallaxController() {
  useEffect(() => {
    if (typeof window === "undefined") return;
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    if (mq.matches) return;

    const root = document.documentElement;
    let ticking = false;

    const update = () => {
      const scrollY = window.scrollY;
      root.style.setProperty("--parallax-slow", `${scrollY * -0.05}px`);
      root.style.setProperty("--parallax-fast", `${scrollY * -0.12}px`);
      ticking = false;
    };
    const onScroll = () => {
      if (!ticking) {
        ticking = true;
        requestAnimationFrame(update);
      }
    };

    update();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => {
      window.removeEventListener("scroll", onScroll);
      root.style.removeProperty("--parallax-slow");
      root.style.removeProperty("--parallax-fast");
    };
  }, []);

  return null;
}
