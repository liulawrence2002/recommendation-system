"use client";

import type { Book, ModelKind, Rec } from "@/lib/types";
import { MODEL_LABELS } from "@/lib/recommend";
import { formatNumber, formatScore, formatYear } from "@/lib/format";

function Cover({ book, title }: { book?: Book; title: string }) {
  const cover = book?.cover || book?.image || "";
  const url = book?.url || "";
  const inner = cover.startsWith("http") ? (
    <img className="rec-cover" src={cover} alt={title} loading="lazy" referrerPolicy="no-referrer" />
  ) : (
    <span className="rec-cover cover-empty" aria-hidden="true">
      📖
    </span>
  );
  if (url.startsWith("http")) {
    return (
      <a className="cover-link" href={url} target="_blank" rel="noopener noreferrer">
        {inner}
      </a>
    );
  }
  return inner;
}

export default function CandidateList({
  recs,
  bookById,
  model,
}: {
  recs: Rec[] | null;
  bookById: Map<number, Book>;
  model: ModelKind;
}) {
  return (
    <section id="candidates">
      <div className="section-head" style={{ marginBottom: "1.2rem" }}>
        <div className="eyebrow">Candidate list</div>
        <h2 className="section-title" style={{ fontSize: "1.5rem" }}>
          The model-produced shortlist
        </h2>
        <p className="section-copy" style={{ marginTop: "0.4rem" }}>
          The chat below reorders these books without inventing new titles.
        </p>
      </div>

      {recs === null ? (
        <div className="empty-state">
          <div>
            <div className="empty-title">No candidates yet</div>
            <div>Use the filters above to generate a ranked list.</div>
          </div>
        </div>
      ) : recs.length === 0 ? (
        <div className="empty-state">
          <div>
            <div className="empty-title">No candidates matched</div>
            <div>Your filters removed every candidate — relax a filter or lower the minimum ratings.</div>
          </div>
        </div>
      ) : (
        <>
          <p className="section-copy" style={{ marginTop: 0, marginBottom: "0.6rem" }}>
            {MODEL_LABELS[model]} scored {formatNumber(recs.length)} books matching your filters.
          </p>
          <div className="rec-list">
            {recs.map((r, i) => {
              const book = bookById.get(r.book_id);
              return (
                <article className="rec-card" key={r.book_id}>
                  <div className="rank">{i + 1}</div>
                  <Cover book={book} title={r.title} />
                  <div>
                    <div className="rec-title">
                      {book?.url ? (
                        <a href={book.url} target="_blank" rel="noopener noreferrer">
                          {r.title}
                        </a>
                      ) : (
                        r.title
                      )}
                    </div>
                    <div className="rec-author">{r.authors}</div>
                    <div className="rec-meta">
                      <span>{formatYear(book?.year)}</span>
                      <span>Avg {book ? book.avgRating.toFixed(3) : "0.000"}</span>
                      <span>{formatNumber(book?.goodreadsCount)} ratings</span>
                    </div>
                  </div>
                  <div className="rec-score">Score {formatScore(r.score)}</div>
                </article>
              );
            })}
          </div>
        </>
      )}
    </section>
  );
}
