"use client";

import { motion, useReducedMotion } from "framer-motion";
import type { ReactNode } from "react";

/** A fluid scroll-reveal: content drifts up and fades in as it enters view.
    The Whisperflow/Oura calm-motion primitive used across the landing.

    Honors prefers-reduced-motion (renders content statically, never hidden) and
    uses viewport amount:0 so elements already on screen reveal immediately —
    content is never left stuck at opacity:0. */
export default function Reveal({
  children,
  delay = 0,
  y = 22,
  className,
}: {
  children: ReactNode;
  delay?: number;
  y?: number;
  className?: string;
}) {
  const reduce = useReducedMotion();
  if (reduce) return <div className={className}>{children}</div>;
  return (
    <motion.div
      className={className}
      initial={{ opacity: 0, y }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0 }}
      transition={{ duration: 0.7, delay, ease: [0.2, 0.8, 0.2, 1] }}
    >
      {children}
    </motion.div>
  );
}
