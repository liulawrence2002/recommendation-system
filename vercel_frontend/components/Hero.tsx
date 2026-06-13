"use client";

import Link from "next/link";
import { motion, useReducedMotion } from "framer-motion";
import FlippingBook from "./FlippingBook";
import { formatCompact, formatNumber } from "@/lib/format";

const fade = {
  hidden: { opacity: 0, y: 26 },
  show: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: { duration: 0.8, delay: i * 0.12, ease: [0.2, 0.8, 0.2, 1] as const },
  }),
};

export default function Hero({
  users,
  books,
  ratings,
}: {
  users: number;
  books: number;
  ratings: number;
}) {
  const meta = [
    { value: formatNumber(books), label: "Books" },
    { value: formatCompact(ratings), label: "Ratings" },
    { value: formatCompact(users), label: "Readers" },
    { value: "3-stage", label: "Reasoning DAG" },
  ];

  const reduce = useReducedMotion();
  // With reduced motion, render the final state with no entrance animation.

  return (
    <header className="hero">
      <motion.div
        initial={reduce ? false : { opacity: 0, scale: 0.92 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={reduce ? { duration: 0 } : { duration: 1, ease: [0.2, 0.8, 0.2, 1] }}
      >
        <FlippingBook />
      </motion.div>

      <motion.div className="eyebrow hero-kicker" custom={0} variants={fade} initial={reduce ? false : "hidden"} animate="show">
        Collaborative filtering · RAG · conversational re-ranking
      </motion.div>

      <motion.h1 className="hero-title" custom={1} variants={fade} initial={reduce ? false : "hidden"} animate="show">
        Find your next book <span className="glow">by talking it through.</span>
      </motion.h1>

      <motion.p className="hero-copy" custom={2} variants={fade} initial={reduce ? false : "hidden"} animate="show">
        BookRec builds a shortlist with collaborative filtering, then a three-stage
        reasoning pipeline reads your mood, asks when it&apos;s unsure, scores every
        candidate, and re-ranks them — explaining each pick. It never invents a title
        it can&apos;t recommend.
      </motion.p>

      <motion.div className="hero-actions" custom={3} variants={fade} initial={reduce ? false : "hidden"} animate="show">
        <Link className="btn btn-primary" href="/studio">
          Open the studio →
        </Link>
        <a className="btn btn-secondary" href="#how">
          See how it works
        </a>
      </motion.div>

      <motion.div className="hero-meta" custom={4} variants={fade} initial={reduce ? false : "hidden"} animate="show">
        {meta.map((m) => (
          <div className="hero-meta-item" key={m.label}>
            <div className="hero-meta-value">{m.value}</div>
            <div className="hero-meta-label">{m.label}</div>
          </div>
        ))}
      </motion.div>
    </header>
  );
}
