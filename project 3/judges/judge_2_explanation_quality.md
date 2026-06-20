# Judge 2 — Explanation Quality

Scores ONE dimension only: are the per-pick explanations specific and grounded in
the provided book metadata, and tied to the request? It does NOT re-judge whether
the books themselves were the right choice (that is Judge 1's job) — a well-
written reason for a so-so pick can still score well here, and that intentional
split is what lets the two judges disagree on a case.

In Braintrust this is an **LLM-as-a-judge scorer** with the same scale as Judge 1.

## Scale (same for both judges)
- **1.0 — Great:** every explanation names a concrete detail from the book's
  metadata (title/series, author, era, or rating) AND ties it to the request;
  the three reasons are distinct, not interchangeable.
- **0.5 — OK:** explanations are on-topic but partly generic, or only some cite
  real metadata, or two reasons are near-duplicates.
- **0.0 — Poor:** vague, interchangeable, or filler reasons ("a great read"),
  reasons unsupported by the given fields, or claims that contradict the metadata.

## Judge prompt (paste into the scorer)

```
You are evaluating the EXPLANATIONS written by a book re-ranker. You are given
the reader's request, the candidate books (with their metadata), and the
re-ranker's ranked top 3 with one reason each.

Judge ONLY explanation quality. Do NOT judge whether the book choices were
correct — assume the picks are fixed and rate how well each is justified.

A reason is high quality when it:
- cites at least one CONCRETE detail present in that book's metadata (its
  title/series, author, year/era, or average_rating), AND
- connects that detail to the reader's request, AND
- is distinct from the other two reasons (no interchangeable boilerplate).

Penalize generic filler, claims not supported by the provided fields, and
near-duplicate reasons.

Request: {{input.request}}
Candidates (with metadata): {{input.candidates}}
Re-ranker output: {{output}}

Respond with exactly one of: Great, OK, Poor — then one sentence of
justification citing a specific reason from the output.
```

## Choice → score mapping (Braintrust scorer config)
- `Great` → 1
- `OK` → 0.5
- `Poor` → 0
