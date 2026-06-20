# Judge 1 — Ranking Fit

Scores ONE dimension only: do the 3 books that were picked, and their order, fit
the reader's request? It ignores how nicely the explanations are written (that is
Judge 2's job).

In Braintrust this is an **LLM-as-a-judge scorer**. Configure it with the
choice→score mapping below so it returns a number on the shared scale.

## Scale (same for both judges)
- **1.0 — Great:** all 3 picks are reasonable for the request and the ordering is
  defensible; the #1 pick is clearly among the best available matches.
- **0.5 — OK:** roughly half fit (e.g. 1–2 of 3 are good) or the set is fine but
  the ordering is off (a weak pick is ranked above a strong one).
- **0.0 — Poor:** the picks largely ignore the request, recommend off-request
  books, or include a book that was not in the candidate list.

## Judge prompt (paste into the scorer)

```
You are evaluating a book RE-RANKER. You are given the reader's request, the
list of candidate books the re-ranker was allowed to choose from, and the
re-ranker's ranked top 3.

Judge ONLY ranking fit: do the chosen books and their order match the request?
Do NOT reward or penalize the writing quality of the reasons — another judge
handles that.

Consider:
- Does each pick plausibly deliver what the request asks for (its mood/tone and
  its content/character signals)?
- Is the ordering defensible — is the #1 pick among the strongest matches?
- Were all 3 books actually in the candidate list? Any off-list book is an
  automatic Poor.

Request: {{input.request}}
Candidates: {{input.candidates}}
Re-ranker output: {{output}}

Respond with exactly one of: Great, OK, Poor — then one sentence of
justification citing specific books.
```

## Choice → score mapping (Braintrust scorer config)
- `Great` → 1
- `OK` → 0.5
- `Poor` → 0
