/* POST /api/chat — runs the conversational RAG/DAG re-ranker for one user turn.
   Body: { userTurns: string[], candidates: Candidate[], topK, roundsAsked, skipClarify }
   Returns a serialized PipelineResult (picks OR a clarifying question, + trace).

   The browser computes the CF Top-N candidates and posts them here so the LLM
   re-ranks only real, grounded books — exactly the contract from the Streamlit
   app (rag_pipeline never invents titles). Works with or without GEMINI_API_KEY. */

import { NextResponse } from "next/server";
import type { Candidate } from "@/lib/types";
import { runPipeline } from "@/lib/dag/pipeline";
import { getGemini } from "@/lib/gemini";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
// Headroom for up to three sequential Gemini stages (each capped at 15s). Takes
// effect where the plan allows; the route still degrades to the heuristic on any
// per-call timeout. Without a key the heuristic path returns near-instantly.
export const maxDuration = 60;

interface ChatBody {
  userTurns?: unknown;
  candidates?: unknown;
  topK?: unknown;
  roundsAsked?: unknown;
  skipClarify?: unknown;
}

function asStringArray(v: unknown): string[] {
  if (!Array.isArray(v)) return [];
  return v.map((x) => String(x)).filter((s) => s.trim().length > 0);
}

function asCandidates(v: unknown): Candidate[] {
  if (!Array.isArray(v)) return [];
  return v
    .filter((c): c is Record<string, unknown> => typeof c === "object" && c !== null)
    .map((c) => ({
      book_id: Number(c.book_id),
      title: String(c.title ?? ""),
      authors: String(c.authors ?? ""),
      year: (c.year as number | "?") ?? "?",
      average_rating: (c.average_rating as number | "?") ?? "?",
      cf_score: Number(c.cf_score ?? 0),
    }))
    .filter((c) => Number.isFinite(c.book_id));
}

export async function POST(req: Request) {
  let body: ChatBody;
  try {
    body = (await req.json()) as ChatBody;
  } catch {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  const userTurns = asStringArray(body.userTurns);
  const candidates = asCandidates(body.candidates);
  if (userTurns.length === 0) {
    return NextResponse.json({ error: "userTurns is required" }, { status: 400 });
  }

  const topKraw = Number(body.topK);
  const topK = Number.isFinite(topKraw) ? Math.max(1, Math.min(30, Math.round(topKraw))) : 5;
  const roundsRaw = Number(body.roundsAsked);
  const roundsAsked = Number.isFinite(roundsRaw) ? Math.max(0, Math.round(roundsRaw)) : 0;
  const skipClarify = Boolean(body.skipClarify);

  const { llm, hasKey, model } = getGemini();

  try {
    const result = await runPipeline(userTurns, candidates, {
      topK,
      model,
      llm,
      hasKey,
      roundsAsked,
      skipClarify,
    });
    return NextResponse.json(result);
  } catch (err) {
    // runPipeline never throws by contract, but guard the route regardless.
    console.error("chat pipeline error", err);
    return NextResponse.json({ error: "Pipeline failed" }, { status: 500 });
  }
}
