"use client";

import type { StageTrace } from "@/lib/types";
import { matchStrength } from "@/lib/dag/helpers";
import type { ReactNode } from "react";

const STEP_NAMES: Record<string, string> = {
  Intent: "Understand your request",
  Clarify: "Check we have enough",
  Scoring: "Rate every book",
  "Re-rank": "Pick the final list",
};

function safeText(v: unknown, fallback = ""): string {
  if (v === null || v === undefined) return fallback;
  const s = String(v).trim();
  return s || fallback;
}

function asList(v: unknown): unknown[] {
  return Array.isArray(v) ? v : [];
}

function StageBody({ name, output }: { name: string; output: Record<string, unknown> }) {
  if (!output) return null;

  if (name === "Understand your request") {
    const chips: ReactNode[] = [];
    const mood = safeText(output.mood);
    if (mood) chips.push(<span className="trace-chip" key="mood">{mood}</span>);
    for (const key of ["pace", "recency"] as const) {
      const val = safeText(output[key]);
      if (val && val.toLowerCase() !== "any") {
        chips.push(
          <span className="trace-chip" key={key}>
            {key === "pace" ? `${val} pace` : val}
          </span>
        );
      }
    }
    asList(output.genres).forEach((g, i) => {
      const label = safeText(g);
      if (label) chips.push(<span className="trace-chip" key={`g${i}`}>{label}</span>);
    });
    asList(output.themes).slice(0, 3).forEach((t, i) => {
      const label = safeText(t);
      if (label) chips.push(<span className="trace-chip" key={`t${i}`}>{label}</span>);
    });
    asList(output.avoid).forEach((a, i) => {
      const label = safeText(a);
      if (label)
        chips.push(
          <span className="trace-chip avoid" key={`a${i}`}>
            avoid {label}
          </span>
        );
    });
    if (chips.length === 0)
      return <div className="trace-line">Nothing specific yet — we&apos;ll ask a quick question.</div>;
    return <div className="trace-chips">{chips}</div>;
  }

  if (name === "Check we have enough") {
    const q = safeText(output.question);
    if (q) {
      const opts = asList(output.options).map((o) => safeText(o)).filter(Boolean);
      return (
        <>
          <div className="trace-line">
            We asked: <em>&ldquo;{q}&rdquo;</em>
          </div>
          {opts.length > 0 && (
            <div className="trace-chips">
              {opts.map((o, i) => (
                <span className="trace-chip" key={i}>
                  {o}
                </span>
              ))}
            </div>
          )}
        </>
      );
    }
    return <div className="trace-line">Enough detail to go on — moving to the books.</div>;
  }

  if (name === "Rate every book") {
    const scored = asList(output.scored).slice(0, 6) as Record<string, unknown>[];
    const count = Number(output.count ?? scored.length);
    return (
      <>
        <div className="trace-line trace-head">
          We rated {count} books — showing the {scored.length} strongest:
        </div>
        {scored.map((s, i) => {
          const title = safeText(s.title) || `Book #${safeText(s.book_id)}`;
          const strength = matchStrength(s.relevance);
          const reason = safeText(s.reason);
          const flags = asList(s.flags).map((f) => safeText(f)).filter(Boolean);
          return (
            <div className="trace-line" key={i}>
              <span className="trace-strength">{strength}</span>{" "}
              <span className="trace-book">{title}</span>
              {reason && <> — {reason}</>}
              {flags.length > 0 && (
                <span className="trace-flag">
                  {" "}
                  set aside: {flags.map((f) => f.replace("avoid:", "")).join(", ")}
                </span>
              )}
            </div>
          );
        })}
      </>
    );
  }

  if (name === "Pick the final list") {
    const picks = asList(output.picks) as Record<string, unknown>[];
    return (
      <>
        {picks.map((p, i) => {
          const title = safeText(p.title, "Untitled");
          const desc = safeText(p.description);
          const why = safeText(p.explanation);
          return (
            <div className="trace-line" key={i}>
              <span className="trace-rank">{i + 1}</span>{" "}
              <span className="trace-book">{title}</span>
              {desc && <> — {desc}</>}
              {why && <span className="trace-flag"> Why: {why}</span>}
            </div>
          );
        })}
      </>
    );
  }

  return null;
}

export default function TraceView({ trace }: { trace: StageTrace[] | undefined }) {
  if (!trace || trace.length === 0) return null;
  return (
    <details className="reasoning-trace">
      <summary>How we chose these · {trace.length} steps</summary>
      <div className="trace-intro">
        A quick look at how BookRec went from your message to this shortlist.
      </div>
      <div className="trace-stages">
        {trace.map((stage, i) => {
          const name = STEP_NAMES[stage.name] ?? stage.name;
          const body = <StageBody name={name} output={stage.output ?? {}} />;
          return (
            <div className="trace-stage" key={i}>
              <div className="trace-stage-head">
                <span className="trace-step-no">{i + 1}</span>
                <span className="trace-stage-name">{name}</span>
                <span className={`stage-badge ${stage.used_llm ? "live" : "heuristic"}`}>
                  <span className="status-dot" />
                  {stage.used_llm ? "AI" : "Smart rules"}
                </span>
              </div>
              <div className="trace-note">{stage.note}</div>
              <div className="trace-body">{body}</div>
            </div>
          );
        })}
      </div>
    </details>
  );
}
