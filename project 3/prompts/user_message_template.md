# Shared user-message template

Use the SAME user message for Prompt A and Prompt B so the only thing that
changes between the two experiments is the system instruction. Paste this into
the Braintrust prompt's **User** message. The `{{...}}` are Braintrust mustache
variables that pull from each dataset row.

If your dataset row stores `input.request` and `input.candidates` (the JSONL
format), use:

```
Reader's request: {{input.request}}

Candidate books (choose ONLY from these, by their book_id):
{{input.candidates}}

Return the ranked top 3 as specified.
```

If you uploaded the flat CSV (columns `request`, `candidates_json`), use
`{{request}}` and `{{candidates_json}}` instead.

---

## Notes
- `candidates` is a list of objects with: `book_id, title, authors, year,
  average_rating, cf_score` — the exact fields the Project 2 re-ranker uses.
- Keep `temperature = 0` (or as low as the provider allows) for both prompts so
  the comparison is about the instructions, not sampling noise.
- Pick one generation model and use it for BOTH prompts.
