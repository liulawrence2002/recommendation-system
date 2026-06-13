"use client";

import type { Stats } from "@/lib/types";
import { formatNumber, formatPercent } from "@/lib/format";

function Bars({
  data,
  alt = false,
}: {
  data: { label: string; value: number }[];
  alt?: boolean;
}) {
  const max = Math.max(1, ...data.map((d) => d.value));
  return (
    <div className={data.length > 8 ? "bars-scroll" : ""}>
      <div className="bars" style={{ minWidth: data.length > 8 ? `${data.length * 42}px` : undefined }}>
        {data.map((d) => (
          <div className="bar-col" key={d.label} title={`${d.label}: ${formatNumber(d.value)}`}>
            <div className={`bar${alt ? " alt" : ""}`} style={{ height: `${(d.value / max) * 100}%` }} />
            <div className="bar-label">{d.label}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function Eda({ eda }: { eda: Stats["eda"] }) {
  const ratingData = eda.ratingDist.map((r) => ({
    label: r.rating.toFixed(0),
    value: r.count,
  }));
  const decadeData = eda.decadeDist.map((d) => ({ label: d.decade, value: d.count }));

  return (
    <details className="card card-pad" id="insights">
      <summary className="disclosure-summary">Dataset insights — users, books &amp; ratings (EDA)</summary>
      <div style={{ marginTop: "1.4rem" }}>
        <div className="eda-grid">
          <div>
            <div className="chart-caption">How readers rate — rating-value distribution</div>
            <Bars data={ratingData} />
          </div>
          <div>
            <div className="chart-caption">Catalog by publication decade</div>
            <Bars data={decadeData} alt />
          </div>
        </div>

        <div className="chart-caption" style={{ marginTop: "1.8rem" }}>
          Most-rated books — the head of the popularity long tail
        </div>
        <div style={{ overflowX: "auto" }}>
          <table className="table">
            <thead>
              <tr>
                <th>Title</th>
                <th>Author</th>
                <th className="num">Ratings</th>
                <th className="num">Avg</th>
              </tr>
            </thead>
            <tbody>
              {eda.topBooks.map((b, i) => (
                <tr key={i}>
                  <td>{b.title}</td>
                  <td>{b.authors}</td>
                  <td className="num">{formatNumber(b.ratings)}</td>
                  <td className="num">{b.avgRating.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="prose" style={{ marginTop: "1.8rem" }}>
          <p>
            <strong>What the data shows</strong>
          </p>
          <ul>
            <li>
              <strong>Positivity bias.</strong> The mean rating is{" "}
              <strong>{eda.meanRating.toFixed(2)} / 5</strong> and{" "}
              <strong>{formatPercent(eda.pct4plus, 0)}</strong> of ratings are 4★ or higher —
              people mostly log books they already liked, which makes a popularity baseline
              hard to beat.
            </li>
            <li>
              <strong>Popularity is a long tail.</strong> The top 10% most-rated books capture{" "}
              <strong>{formatPercent(eda.top10pctShare, 0)}</strong> of all ratings, while the
              median book has only <strong>{eda.perBookMedian}</strong> ratings.
            </li>
            <li>
              <strong>Active vs. casual readers.</strong> The median reader has rated{" "}
              <strong>{eda.perUserMedian}</strong> books (mean {eda.perUserMean.toFixed(0)}, max{" "}
              {formatNumber(eda.perUserMax)}) — a classic power-user skew.
            </li>
            <li>
              <strong>Sparsity.</strong> The reader × book matrix is{" "}
              <strong>{formatPercent(eda.sparsity)}</strong> empty — the core challenge for
              collaborative filtering (cold-start, thin neighbourhoods).
            </li>
          </ul>
          <p>
            <strong>Why it matters:</strong> strong positivity + popularity concentration mean a
            popularity/mean baseline is a tough benchmark, and sparsity limits neighbourhood CF —
            exactly why we benchmark UBCF/IBCF against the baseline and add a language layer for
            personalization <em>beyond</em> popularity.
          </p>
        </div>
      </div>
    </details>
  );
}
