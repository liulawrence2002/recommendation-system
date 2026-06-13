"use client";

import type { Book, AssistantMessage, RerankedPick } from "@/lib/types";
import IntentChips from "./IntentChips";
import TraceView from "./TraceView";

function PickCover({ book }: { book?: Book }) {
  const cover = book?.cover || book?.image || "";
  const url = book?.url || "";
  const inner = cover.startsWith("http") ? (
    <img className="pick-cover-img" src={cover} alt="" loading="lazy" referrerPolicy="no-referrer" />
  ) : (
    <span className="pick-cover-img pick-cover-empty" aria-hidden="true">
      📖
    </span>
  );
  if (url.startsWith("http")) {
    return (
      <a className="pick-cover" href={url} target="_blank" rel="noopener noreferrer">
        {inner}
      </a>
    );
  }
  return <div className="pick-cover">{inner}</div>;
}

function PickCard({ pick, rank, book }: { pick: RerankedPick; rank: number; book?: Book }) {
  const title = pick.title || "Untitled";
  return (
    <article className="pick-card">
      <div className="pick-rank">{rank}</div>
      <PickCover book={book} />
      <div className="pick-title">
        {book?.url ? (
          <a href={book.url} target="_blank" rel="noopener noreferrer">
            {title}
          </a>
        ) : (
          title
        )}
        <span className="pick-author"> by {pick.authors || "Unknown author"}</span>
      </div>
      {pick.description && <div className="pick-desc">{pick.description}</div>}
      <div className="pick-why">
        <span className="pick-why-label">Why #{rank}</span>
        {pick.explanation || "Ranked for this preference."}
      </div>
    </article>
  );
}

export default function AssistantBubble({
  message,
  bookById,
}: {
  message: AssistantMessage;
  bookById: Map<number, Book>;
}) {
  if (message.kind === "clarify") {
    return (
      <div className="msg msg-assistant">
        <div className="avatar">B</div>
        <div className="assistant-body">
          <div className="assistant-name">BookRec</div>
          <div className="assistant-lead">
            {message.question_text || "Could you tell me a bit more?"}
          </div>
          <IntentChips intent={message.intent} />
          <div className="clarify-hint">
            Choose an option below, type your own answer, or skip straight to recommendations.
          </div>
          <TraceView trace={message.trace} />
          {message.source && <div className="assistant-meta">Source: {message.source}</div>}
        </div>
      </div>
    );
  }

  const lead =
    message.intent?.summary ||
    `${message.refine ? "Refined the shortlist" : "Here is your shortlist"} for “${
      message.pref || "your request"
    }”.`;

  return (
    <div className="msg msg-assistant">
      <div className="avatar">B</div>
      <div className="assistant-body">
        <div className="assistant-name">BookRec</div>
        <div className="assistant-lead">{lead}</div>
        <IntentChips intent={message.intent} />
        {message.picks.length > 0 ? (
          <div className="pick-list">
            {message.picks.map((p, i) => (
              <PickCard key={p.book_id} pick={p} rank={i + 1} book={bookById.get(p.book_id)} />
            ))}
          </div>
        ) : (
          <div className="pick-empty">
            No candidates matched closely enough. Try regenerating candidates or loosening the
            filters above.
          </div>
        )}
        <TraceView trace={message.trace} />
        {message.source && <div className="assistant-meta">Source: {message.source}</div>}
      </div>
    </div>
  );
}
