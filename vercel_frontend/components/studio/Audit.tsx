"use client";

import type { Stats } from "@/lib/types";

export default function Audit({ audit }: { audit: Stats["audit"] }) {
  return (
    <details className="card card-pad" id="quality">
      <summary className="disclosure-summary">Model quality &amp; audit</summary>
      <div style={{ marginTop: "1.2rem" }}>
        <p className="prose" style={{ marginBottom: "1rem" }}>
          <strong>Offline hold-out from the project notebooks</strong> (fixed test set, seed{" "}
          {audit.seed}). User-based CF with Pearson similarity is the strongest model on
          Precision/Recall/F1@{audit.k}.
        </p>
        <table className="table">
          <thead>
            <tr>
              <th>Model</th>
              <th className="num">P@{audit.k}</th>
              <th className="num">R@{audit.k}</th>
              <th className="num">F1@{audit.k}</th>
            </tr>
          </thead>
          <tbody>
            {audit.notebookMetrics.map((m) => (
              <tr key={m.model}>
                <td>{m.model}</td>
                <td className="num">{m.p10.toFixed(4)}</td>
                <td className="num">{m.r10.toFixed(4)}</td>
                <td className="num">{m.f1.toFixed(4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="prose" style={{ marginTop: "1rem" }}>
          The margins are tight — on sparse, popularity-skewed data, collaborative filtering only
          narrowly edges the baseline. That honest finding is exactly why a language re-ranking
          layer earns its place: it personalizes <em>beyond</em> what ratings alone express.
        </p>
      </div>
    </details>
  );
}
