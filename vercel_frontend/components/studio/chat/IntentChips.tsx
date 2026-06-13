"use client";

import type { Intent } from "@/lib/types";

/** Pills summarizing the DAG's extracted intent — port of app.py render_intent_chips. */
export default function IntentChips({ intent }: { intent: Intent | null }) {
  if (!intent) return null;
  const chips: { text: string; avoid?: boolean }[] = [];
  if (intent.mood) chips.push({ text: intent.mood });
  if (intent.pace && intent.pace.toLowerCase() !== "any") chips.push({ text: `${intent.pace} pace` });
  if (intent.recency && intent.recency.toLowerCase() !== "any") chips.push({ text: intent.recency });
  for (const g of (intent.genres ?? []).slice(0, 3)) if (g) chips.push({ text: g });
  for (const t of (intent.themes ?? []).slice(0, 2)) if (t) chips.push({ text: t });
  for (const a of (intent.avoid ?? []).slice(0, 3)) if (a) chips.push({ text: `no ${a}`, avoid: true });

  if (chips.length === 0) return null;
  return (
    <div className="intent-row">
      {chips.map((c, i) => (
        <span className={`intent-chip${c.avoid ? " avoid" : ""}`} key={i}>
          {c.text}
        </span>
      ))}
    </div>
  );
}
