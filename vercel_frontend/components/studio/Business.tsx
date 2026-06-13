"use client";

export default function Business() {
  return (
    <details className="card card-pad" id="business">
      <summary className="disclosure-summary">Business applications &amp; recommended approach</summary>
      <div className="prose" style={{ marginTop: "1.2rem" }}>
        <p>
          <strong>Collaborative filtering (UBCF / IBCF).</strong> Powers the &ldquo;readers like
          you also enjoyed…&rdquo; experience — discovery, engagement, retention, cross-sell. Cheap
          to serve once trained and needs no content metadata, just the behaviour signal the
          business already collects.
        </p>
        <p>
          <strong>Language re-ranking layer.</strong> Turns a static Top-N into mood-aware,
          natural-language personalization with a short, explainable reason per pick. It captures
          intent (&ldquo;a cozy mystery, nothing gory&rdquo;) that ratings can&apos;t express, and
          differentiates the experience.
        </p>
        <p>
          <strong>Challenges to plan for.</strong>
        </p>
        <ul>
          <li>
            <em>Collaborative filtering:</em> cold-start for new users/books, data sparsity,
            popularity bias, scaling similarity to millions of users, and the gap between offline
            metrics and real online lift.
          </li>
          <li>
            <em>Language layer:</em> API cost &amp; latency at scale, grounding (must re-rank only
            real candidates), prompt-injection/safety, key management, and measuring incremental
            value over plain CF.
          </li>
        </ul>
        <p>
          <strong>Recommended approach.</strong> Use UBCF (Pearson) to generate candidates, keep
          the popularity/mean baseline as a guardrail and cold-start fallback, and layer the
          re-ranker strictly on top of CF candidates for personalization + explanations. Cache and
          rate-limit LLM calls to control cost, and graduate from offline Precision/Recall@K to
          A/B-tested online lift once live.
        </p>
        <p className="config-note">
          Model for the language layer: Google Gemini (<code>gemini-2.0-flash</code>). The key is
          read from the environment and never committed; without it the app falls back to a
          transparent heuristic re-ranker so it always runs.
        </p>
      </div>
    </details>
  );
}
