"use client";

import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import type {
  ChatMessage,
  ModelKind,
  PipelineResult,
  Rec,
  RecommendMessage,
  ShelfItem,
} from "@/lib/types";
import { loadDataset, type Dataset as DatasetType } from "@/lib/clientData";
import {
  candidatesFromRecs,
  filterBookIds,
  recommendTopN,
  MODEL_LABELS,
} from "@/lib/recommend";
import FilterPanel from "./FilterPanel";
import Shelf from "./Shelf";
import Chat from "./Chat";
import IntroGate from "./IntroGate";

export default function StudioApp() {
  const [data, setData] = useState<DatasetType | null>(null);
  const [loadError, setLoadError] = useState(false);

  // filter / model controls
  const [authors, setAuthors] = useState<string[]>([]);
  const [decades, setDecades] = useState<string[]>([]);
  const [model, setModel] = useState<ModelKind>("ubcf");
  const [minRatings, setMinRatings] = useState(20);
  const [candidateCount, setCandidateCount] = useState(10);

  // generated candidates
  const [recs, setRecs] = useState<Rec[] | null>(null);
  const [generatedKey, setGeneratedKey] = useState<string | null>(null);

  // chat
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [pending, setPending] = useState(false);
  const [depth, setDepth] = useState<"Focused" | "Extended">("Focused");
  const [chatError, setChatError] = useState<string | null>(null);

  // Monotonic request token: bumping it invalidates any in-flight /api/chat
  // response so a slow reply can never resurrect a thread that was reset by a
  // filter change, a regenerate, or a manual "new chat".
  const genRef = useRef(0);

  useEffect(() => {
    loadDataset().then(setData).catch(() => setLoadError(true));
  }, []);

  const configKey = useMemo(
    () =>
      JSON.stringify([
        model,
        minRatings,
        candidateCount,
        [...authors].sort(),
        [...decades].sort(),
      ]),
    [model, minRatings, candidateCount, authors, decades]
  );

  // When any control changes after generating, the shortlist is stale: clear it
  // and the conversation (mirrors the Streamlit app's regenerate-to-refresh flow).
  useEffect(() => {
    if (generatedKey !== null && configKey !== generatedKey) {
      genRef.current++; // invalidate any in-flight chat request
      setRecs(null);
      setGeneratedKey(null);
      setMessages([]);
      setPending(false);
      setChatError(null);
    }
  }, [configKey, generatedKey]);

  function generate() {
    if (!data) return;
    genRef.current++; // a fresh shortlist supersedes any in-flight chat request
    const allowed = filterBookIds(data.books, { authors, decades });
    const next = recommendTopN(data.books, data.scores, {
      model,
      topN: candidateCount,
      minRatings,
      allowedIds: allowed,
    });
    setRecs(next);
    setGeneratedKey(configKey);
    setMessages([]);
    setPending(false);
    setChatError(null);
  }

  async function submit(text: string, skipClarify = false) {
    if (!data || recs === null) return;
    const topK = depth === "Focused" ? 5 : 8;
    const roundsAsked = messages.filter(
      (m) => m.role === "assistant" && m.kind === "clarify"
    ).length;
    // A turn is a "refine" only if a real shortlist has already been delivered —
    // not merely because the thread has >1 user turn (answering a clarifying
    // question adds a turn without making the next result a refinement).
    const priorRecs = messages.filter(
      (m) => m.role === "assistant" && m.kind === "recommend"
    ).length;

    const myGen = ++genRef.current; // claim this request; supersede any earlier one
    const withUser: ChatMessage[] = [...messages, { role: "user", content: text }];
    setMessages(withUser);
    setPending(true);
    setChatError(null);

    const userTurns = withUser
      .filter((m): m is { role: "user"; content: string } => m.role === "user")
      .map((m) => m.content);
    const candidates = candidatesFromRecs(recs, data.bookById);

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ userTurns, candidates, topK, roundsAsked, skipClarify }),
      });
      if (genRef.current !== myGen) return; // superseded mid-flight — drop the result
      if (!res.ok) throw new Error(`status ${res.status}`);
      const result = (await res.json()) as PipelineResult;
      if (genRef.current !== myGen) return;

      if (result.question) {
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            kind: "clarify",
            question_text: result.question!.prompt,
            options: result.question!.options ?? [],
            source: result.source,
            used_llm: result.used_llm,
            intent: result.intent,
            trace: result.trace,
          },
        ]);
      } else {
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            kind: "recommend",
            picks: result.picks,
            source: result.source,
            used_llm: result.used_llm,
            refine: priorRecs > 0,
            pref: userTurns[0] ?? text,
            intent: result.intent,
            trace: result.trace,
            scored: result.scored, // for the shelf "% match" badge (UI-only)
          },
        ]);
      }
    } catch {
      if (genRef.current !== myGen) return;
      setChatError("Something went wrong reaching the recommender. Please try again.");
    } finally {
      if (genRef.current === myGen) setPending(false);
    }
  }

  function resetChat() {
    genRef.current++; // invalidate any in-flight chat request
    setMessages([]);
    setPending(false);
    setChatError(null);
  }

  // The shelf shows the re-ranked picks (with a % match from the pipeline's
  // per-book relevance) once a recommend turn exists, otherwise the raw CF list.
  const lastRecommend = useMemo<RecommendMessage | null>(() => {
    for (let i = messages.length - 1; i >= 0; i--) {
      const m = messages[i];
      if (m.role === "assistant" && m.kind === "recommend") return m;
    }
    return null;
  }, [messages]);

  const shelfItems = useMemo<ShelfItem[] | null>(() => {
    if (!data || recs === null) return null;
    const bookById = data.bookById;
    if (lastRecommend) {
      const relById = new Map(
        (lastRecommend.scored ?? []).map((s) => [s.book_id, s.relevance])
      );
      return lastRecommend.picks.map((p, i) => {
        const rel = relById.get(p.book_id);
        return {
          book: bookById.get(p.book_id),
          book_id: p.book_id,
          title: p.title,
          authors: p.authors,
          rank: i + 1,
          match: rel != null ? Math.round(100 * rel) : null,
          why: p.explanation || null,
          synopsis: p.description || null,
          cfScore: null,
        };
      });
    }
    return recs.map((r, i) => ({
      book: bookById.get(r.book_id),
      book_id: r.book_id,
      title: r.title,
      authors: r.authors,
      rank: i + 1,
      match: null,
      why: null,
      synopsis: null,
      cfScore: r.score,
    }));
  }, [data, recs, lastRecommend]);

  function scrollToChat() {
    const el = document.getElementById("chat");
    if (el) {
      const y = el.getBoundingClientRect().top + window.scrollY - 70;
      window.scrollTo({ top: y, behavior: "smooth" });
    }
    setTimeout(() => {
      document.querySelector<HTMLTextAreaElement>("#chat textarea")?.focus();
    }, 420);
  }

  let inner: ReactNode;
  if (loadError) {
    inner = (
      <div className="container" style={{ paddingTop: "8rem" }}>
        <div className="empty-state">
          <div>
            <div className="empty-title">Couldn&apos;t load the dataset</div>
            <div>
              Run <code>python scripts/export_data.py</code> to generate the data files in
              <code> public/data</code>, then reload.
            </div>
          </div>
        </div>
      </div>
    );
  } else if (!data) {
    inner = (
      <div className="studio-loading">
        <div className="spinner" />
        <div>Loading the catalog &amp; pre-computed model scores…</div>
      </div>
    );
  } else {
    inner = (
      <div className="container stack" style={{ paddingTop: "1rem" }}>
        <FilterPanel
          authorOptions={data.stats.filters.authors}
          decadeOptions={data.stats.filters.decades}
          selectedAuthors={authors}
          selectedDecades={decades}
          model={model}
          minRatings={minRatings}
          candidateCount={candidateCount}
          autoUser={data.scores.autoUser}
          autoUserRatings={data.scores.autoUserRatings}
          onAuthors={setAuthors}
          onDecades={setDecades}
          onModel={setModel}
          onMinRatings={setMinRatings}
          onCandidateCount={setCandidateCount}
          onGenerate={generate}
        />

        <Shelf
          items={shelfItems}
          modelLabel={MODEL_LABELS[model]}
          reranked={lastRecommend !== null}
          pending={pending}
          onRefine={scrollToChat}
        />

        {chatError && <div className="chat-lock">{chatError}</div>}
        <Chat
          messages={messages}
          pending={pending}
          hasRecs={recs !== null}
          bookById={data.bookById}
          depth={depth}
          onDepth={setDepth}
          onSubmit={submit}
          onReset={resetChat}
        />
      </div>
    );
  }

  return <IntroGate>{inner}</IntroGate>;
}
