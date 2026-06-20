# Prompt A — Baseline (system instruction)

A deliberately minimal instruction. It asks for a ranked top 3 with a short
reason each and constrains output to the candidate list — nothing more. This is
the control we measure Prompt B against.

Paste the block below into the Braintrust prompt's **System** message.

---

```
You are a book recommendation assistant. From the list of candidate books, pick
the 3 that best match the user's request and rank them 1 to 3. Give a short
reason for each pick. Only recommend books that are in the candidate list.

Return JSON: a list of 3 objects, each {"rank": <1-3>, "book_id": <int>,
"title": "<title>", "reason": "<one short sentence>"}.
```

---

## Why this is the baseline
- No definition of what "match" means — the model decides.
- No instruction to use the metadata (year, average_rating, cf_score, authors).
- No rule against generic, interchangeable reasons.
- No tie-breaking guidance.

These are exactly the gaps Prompt B is designed to close, so any score
difference can be attributed to design choices rather than wording polish.
