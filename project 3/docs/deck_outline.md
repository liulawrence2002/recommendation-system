# Deck outline — Project 3 (≤ 8 slides + ≤ 2 appendix, font ≥ 16, submit PDF)

Build from this skeleton once your Braintrust run is done. Each slide lists its
job and the evidence to drop in.

### Slide 1 — Title
- Project title: "Evaluating the LLM Re-Ranker (Goodreads)".
- **Team number + all member names** (must match your Project 2 submission type:
  individual stays individual, group stays group).
- One line: "Comparing two re-ranking prompts with two LLM judges in Braintrust."

### Slide 2 — Evaluation dataset & setup
- 10 users × 2 contrasting requests = **20 cases**; one case = one row.
- Requests: mood/tone vs. content/character (quote both).
- Candidates: Top-10 from **UBCF (pearson_baseline, k=25)** — the shipped CF model.
- Fields the re-ranker sees: `book_id, title, authors, year, average_rating,
  cf_score`.
- Mention the `cf_score` saturation quirk and why it makes the test fair.
- *Evidence:* screenshot of the Braintrust dataset grid (or put it in appendix).

### Slide 3 — The two prompts
- Prompt A (baseline): minimal — pick & rank top 3, short reason, candidates only.
- Prompt B (designed): explicit fit criteria, cf_score demoted to tie-breaker,
  every reason must cite metadata, anti-duplication rule, persona.
- Use the design-difference table from `prompts/prompt_B_designed.md`.
- *Evidence:* screenshots of both prompts (or appendix).

### Slide 4 — The two judges
- Ranking Fit: do the 3 picks + order match the request? 1 / 0.5 / 0.
- Explanation Quality: are reasons specific + grounded in metadata? 1 / 0.5 / 0.
- State the model used for judging and whether it differs from the generator.
- *Evidence:* screenshots of both scorer configs (or appendix).

### Slide 5 — Score comparison (the headline result)
- Table: average Ranking Fit and Explanation Quality for Prompt A vs. Prompt B.
- Optionally split by request type (mood vs. character).
- One sentence on the size/direction of the gap.
- *Evidence:* screenshot of the Experiment results/comparison view.

### Slide 6 — Trace deep-dive
- One case where **Prompt B clearly helped** (show both outputs + judge reasons).
- One case where **Prompt B did not help / regressed**.
- One case where **the two judges disagree** (e.g. good fit, weak explanation) —
  explain why that split is informative.
- *Evidence:* trace screenshots with the judge justifications visible.

### Slide 7 — Evaluation design (REQUIRED dedicated slide)
- **Limitations of a single LLM judge:** self-preference / same-model bias,
  prompt-sensitivity and non-determinism, no ground truth, scale compression
  (1/0.5/0 is coarse), 20 cases = small sample, judge may reward verbosity.
- **Concrete improvements:** judge model ≠ generator; multiple judges + majority
  or averaging; human spot-check / calibration set; few-shot anchored rubrics;
  more cases and more diverse requests; randomize output order to fight position
  bias; report inter-judge agreement.
- **Other evaluation types relevant to re-ranking beyond an LLM judge:** offline
  ranking metrics (NDCG, MAP, Precision/Recall@k against held-out likes),
  diversity/novelty/coverage, list-level redundancy, latency & cost per request,
  constraint checks (no off-list books, valid JSON), and online A/B tests
  (CTR, saves, session length).

### Slide 8 — Recommendation (business terms)
- Which prompt you'd ship and why, tied to **cost vs. benefit**: Prompt B's
  longer instruction costs more tokens per call — is the quality lift worth it at
  your request volume? Where does B matter most (which request type)?
- A clear, one-line ship decision and the condition that would change it.

### Appendix (optional, ≤ 2 slides)
- A1: full Braintrust evidence — dataset grid, both prompt configs, both judge
  configs at full size.
- A2: extra traces / the per-request-type score breakdown table.
