"use client";

import { useRef, useState } from "react";
import type { Book, ChatMessage, ClarifyMessage } from "@/lib/types";
import AssistantBubble from "./chat/Messages";

const STARTERS = [
  "Something fast-paced and adventurous",
  "A thoughtful literary list with emotional depth",
  "Dark, mysterious books with strong atmosphere",
];

const REFINERS = [
  "Make them darker and moodier",
  "Lean more recent",
  "Lighter and funnier",
  "Surprise me with a wildcard",
];

export default function Chat({
  messages,
  pending,
  hasRecs,
  bookById,
  depth,
  onDepth,
  onSubmit,
  onReset,
}: {
  messages: ChatMessage[];
  pending: boolean;
  hasRecs: boolean;
  bookById: Map<number, Book>;
  depth: "Focused" | "Extended";
  onDepth: (d: "Focused" | "Extended") => void;
  onSubmit: (text: string, skipClarify?: boolean) => void;
  onReset: () => void;
}) {
  const [text, setText] = useState("");
  const taRef = useRef<HTMLTextAreaElement>(null);

  const refining = messages.length > 0;
  const last = messages[messages.length - 1];
  const awaitingClarify =
    !pending && last?.role === "assistant" && last.kind === "clarify";
  const clarifyOptions = awaitingClarify
    ? ((last as ClarifyMessage).options ?? []).filter((o) => o.trim())
    : [];
  const disabled = pending || !hasRecs;

  function send(value: string, skipClarify = false) {
    const v = value.trim();
    if (!v || disabled) return;
    onSubmit(v, skipClarify);
    setText("");
    if (taRef.current) taRef.current.style.height = "auto";
  }

  function onTextareaInput(e: React.FormEvent<HTMLTextAreaElement>) {
    const el = e.currentTarget;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 256)}px`;
  }

  return (
    <section className="chat" id="chat">
      {refining ? (
        <div className="thread">
          {messages.map((m, i) =>
            m.role === "user" ? (
              <div className="msg msg-user" key={i}>
                <div className="bubble-user">{m.content}</div>
              </div>
            ) : (
              <AssistantBubble key={i} message={m} bookById={bookById} />
            )
          )}
          {pending && (
            <div className="msg msg-assistant">
              <div className="avatar">B</div>
              <div className="assistant-body">
                <div className="assistant-name">BookRec</div>
                <div className="assistant-lead">Working through your request…</div>
                <div className="stage-progress">
                  {["Reading your request", "Rating the books", "Choosing the best"].map((l) => (
                    <span className="stage-pill" key={l}>
                      <span className="stage-pill-dot" />
                      {l}
                    </span>
                  ))}
                </div>
                <div className="thinking">
                  <span />
                  <span />
                  <span />
                </div>
              </div>
            </div>
          )}
        </div>
      ) : (
        <div className="chat-hero">
          <h2 className="chat-title">Ready when you are.</h2>
          <p className="chat-subtitle">
            Ask BookRec for a mood, genre, theme, or reading goal — then keep refining. It
            personalizes the candidates above and never invents books outside the shortlist.
          </p>
        </div>
      )}

      {!hasRecs && (
        <div className="chat-lock">
          Generate filtered candidates first so the chat can re-rank real books.
        </div>
      )}

      <div className="composer">
        <textarea
          ref={taRef}
          value={text}
          rows={1}
          aria-label="Message BookRec"
          placeholder={
            refining
              ? "Refine your shortlist — adjust tone, pace, setting, tropes to avoid, or steer somewhere new…"
              : "Ask for dark academia, a cozy mystery, fast-paced sci-fi, or whatever mood you're in…"
          }
          disabled={disabled}
          onChange={(e) => setText(e.target.value)}
          onInput={onTextareaInput}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              // Don't submit mid-IME-composition (e.g. committing a CJK candidate).
              if (e.nativeEvent.isComposing || e.keyCode === 229) return;
              e.preventDefault();
              send(text);
            }
          }}
        />
        <div className="composer-row">
          <div className="composer-left">
            <select
              className="select depth-select"
              value={depth}
              onChange={(e) => onDepth(e.target.value as "Focused" | "Extended")}
              aria-label="Depth"
              title="Focused returns 5 picks; Extended returns 8."
            >
              <option value="Focused">Focused · 5</option>
              <option value="Extended">Extended · 8</option>
            </select>
          </div>
          <button
            className="send-btn"
            onClick={() => send(text)}
            disabled={disabled || !text.trim()}
            aria-label="Send"
          >
            ↑
          </button>
        </div>
      </div>

      {/* suggestion area */}
      {awaitingClarify ? (
        <>
          {clarifyOptions.length > 0 && (
            <div className="suggestions">
              {clarifyOptions.map((o) => (
                <button className="suggestion" key={o} disabled={disabled} onClick={() => send(o)}>
                  {o}
                </button>
              ))}
            </div>
          )}
          <div className="suggestions" style={{ marginTop: "0.6rem" }}>
            <button
              className="suggestion"
              disabled={disabled}
              onClick={() => send("Just recommend something", true)}
            >
              Just recommend something →
            </button>
            <button className="suggestion" disabled={pending} onClick={onReset}>
              ↻ Start a new chat
            </button>
          </div>
        </>
      ) : refining ? (
        <>
          <div className="suggestions">
            {REFINERS.map((c) => (
              <button className="suggestion" key={c} disabled={disabled} onClick={() => send(c)}>
                {c}
              </button>
            ))}
          </div>
          <div className="suggestions" style={{ marginTop: "0.6rem" }}>
            <button className="suggestion" disabled={pending} onClick={onReset}>
              ↻ Start a new chat
            </button>
          </div>
        </>
      ) : (
        <div className="suggestions">
          {STARTERS.map((c) => (
            <button className="suggestion" key={c} disabled={!hasRecs} onClick={() => send(c)}>
              {c}
            </button>
          ))}
        </div>
      )}

      <div className="chat-toolbar">
        BookRec only re-ranks the generated candidates — keep refining to steer tone, pace, or
        setting. It never invents books outside the shortlist.
      </div>
    </section>
  );
}
