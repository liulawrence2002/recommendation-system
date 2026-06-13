"use client";

import type { Stats } from "@/lib/types";
import { formatNumber, formatPercent } from "@/lib/format";

export default function Metrics({ headline }: { headline: Stats["headline"] }) {
  const cards = [
    { label: "Readers", value: formatNumber(headline.users), note: "reader profiles" },
    { label: "Books", value: formatNumber(headline.books), note: "catalog items" },
    { label: "Ratings", value: formatNumber(headline.ratings), note: "observed signals" },
    { label: "Sparsity", value: formatPercent(headline.sparsity), note: "matrix empty" },
  ];
  return (
    <div className="metric-grid">
      {cards.map((c) => (
        <div className="metric-card" key={c.label}>
          <div className="metric-label">{c.label}</div>
          <div className="metric-value">{c.value}</div>
          <div className="metric-note">{c.note}</div>
        </div>
      ))}
    </div>
  );
}
