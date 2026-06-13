"use client";

import type { ModelKind } from "@/lib/types";
import { MODEL_LABELS } from "@/lib/recommend";
import { formatNumber } from "@/lib/format";
import Multiselect from "./Multiselect";

const MODEL_ORDER: ModelKind[] = ["ubcf", "ibcf", "baseline", "svd", "popularity"];

export default function FilterPanel({
  authorOptions,
  decadeOptions,
  selectedAuthors,
  selectedDecades,
  model,
  minRatings,
  candidateCount,
  autoUser,
  autoUserRatings,
  onAuthors,
  onDecades,
  onModel,
  onMinRatings,
  onCandidateCount,
  onGenerate,
}: {
  authorOptions: string[];
  decadeOptions: string[];
  selectedAuthors: string[];
  selectedDecades: string[];
  model: ModelKind;
  minRatings: number;
  candidateCount: number;
  autoUser: number;
  autoUserRatings: number;
  onAuthors: (v: string[]) => void;
  onDecades: (v: string[]) => void;
  onModel: (v: ModelKind) => void;
  onMinRatings: (v: number) => void;
  onCandidateCount: (v: number) => void;
  onGenerate: () => void;
}) {
  return (
    <div className="card card-pad" id="filter">
      <h2 className="section-title" style={{ fontSize: "1.35rem" }}>
        Filtering
      </h2>
      <p className="section-copy" style={{ marginTop: "0.4rem", marginBottom: "1.2rem" }}>
        Shape the candidate pool the chat assistant will re-rank. Filter by author and decade;
        model and tuning live under advanced options.
      </p>

      <div className="filter-grid">
        <Multiselect
          label="Authors"
          options={authorOptions}
          selected={selectedAuthors}
          onChange={onAuthors}
          placeholder="Search 5,800+ authors…"
        />
        <Multiselect
          label="Decades"
          options={decadeOptions}
          selected={selectedDecades}
          onChange={onDecades}
          placeholder="Add a decade…"
        />
      </div>

      <details className="advanced">
        <summary className="disclosure-summary">Advanced options</summary>
        <div className="advanced-grid" style={{ marginTop: "1.1rem" }}>
          <div>
            <label className="field-label" htmlFor="model-select">
              Model
            </label>
            <select
              id="model-select"
              className="select"
              value={model}
              onChange={(e) => onModel(e.target.value as ModelKind)}
            >
              {MODEL_ORDER.map((m) => (
                <option key={m} value={m}>
                  {MODEL_LABELS[m]}
                </option>
              ))}
            </select>
          </div>
          <div>
            <div className="slider-row">
              <label className="field-label" htmlFor="cand-slider">
                Candidate count
              </label>
              <span className="slider-value">{candidateCount}</span>
            </div>
            <input
              id="cand-slider"
              className="slider"
              type="range"
              min={5}
              max={30}
              value={candidateCount}
              onChange={(e) => onCandidateCount(Number(e.target.value))}
            />
          </div>
          <div>
            <div className="slider-row">
              <label className="field-label" htmlFor="min-slider">
                Minimum ratings per book
              </label>
              <span className="slider-value">{minRatings}</span>
            </div>
            <input
              id="min-slider"
              className="slider"
              type="range"
              min={0}
              max={200}
              value={minRatings}
              onChange={(e) => onMinRatings(Number(e.target.value))}
            />
          </div>
          <div style={{ alignSelf: "end" }}>
            <span className="reader-pill">
              Reader: most active · #{autoUser} ({formatNumber(autoUserRatings)} ratings)
            </span>
            <div className="config-note">
              CF scores are pre-computed for the most-active reader (UBCF neighbourhood k=21).
              The chat does the personalization on top.
            </div>
          </div>
        </div>
      </details>

      <div className="pill-row">
        <span className="pill">{MODEL_LABELS[model]}</span>
        <span className="pill">Top {candidateCount}</span>
        <span className="pill">Min {formatNumber(minRatings)} book ratings</span>
        {selectedAuthors.length > 0 && (
          <span className="pill">
            {selectedAuthors.length} author{selectedAuthors.length > 1 ? "s" : ""}
          </span>
        )}
        {selectedDecades.length > 0 && <span className="pill">{selectedDecades.join(", ")}</span>}
      </div>

      <button className="btn btn-primary btn-block" style={{ marginTop: "1rem" }} onClick={onGenerate}>
        Generate filtered candidates
      </button>
    </div>
  );
}
