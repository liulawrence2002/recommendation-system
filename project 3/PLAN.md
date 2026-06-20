# Project 3 — Evaluating the LLM Re-Ranker (Plan & Playbook)

This folder contains everything for Project 3: evaluating the Project 2 LLM
re-ranking layer with **Braintrust** and **two LLM judges**, comparing **two
system prompts**, and shipping a recommendation.

---

## 1. What the assignment is actually asking (plain English)

In Project 2 you built a recommender: a collaborative-filtering (CF) model
produces a user's Top-N candidate books, then an LLM **re-ranks** those
candidates to a stated preference and writes a short "why" for each.

Project 3 does **not** add features. It **tests whether a better-written
re-ranking instruction produces better recommendations** — and, crucially, it
builds the *evaluation machinery* to prove it, because there's no single correct
answer to "is this a good recommendation?"

You are comparing **two things** along **two dimensions**:

- **Two prompts:** Prompt A (minimal baseline) vs. Prompt B (carefully designed).
- **Two judges:** *Ranking Fit* (did it pick/order the right 3 books?) and
  *Explanation Quality* (are the reasons specific and grounded in the data?).

Every one of your 20 cases is run through **both prompts** and scored by **both
judges** → 20 cases × 2 prompts × 2 judges = **80 LLM-judge scores**, plus 40
generations. That's the API-cost math to keep in mind before adding cases.

The deliverable is a **PDF slide deck (≤ 8 slides, font ≥ 16)** with Braintrust
screenshots as evidence, a Prompt A vs. B comparison, trace analysis, a
ship recommendation in business terms, and one slide critiquing the evaluation
design itself.

---

## 2. What's already built in this folder

| Path | What it is | Status |
|---|---|---|
| `scripts/build_eval_dataset.py` | Regenerates the dataset from the Project 2 UBCF model | ✅ done |
| `data/eval_dataset.csv` | 20 cases (10 users × 2 requests), candidates as JSON — upload this | ✅ done |
| `data/eval_dataset.jsonl` | Same 20 cases in Braintrust `input`/`metadata` shape | ✅ done |
| `data/candidates_preview.txt` | Human-readable look at each user's 10 candidates | ✅ done |
| `prompts/prompt_A_baseline.md` | Prompt A system instruction + rationale | ✅ done |
| `prompts/prompt_B_designed.md` | Prompt B system instruction + design table | ✅ done |
| `prompts/user_message_template.md` | Shared user message (same for both prompts) | ✅ done |
| `judges/judge_1_ranking_fit.md` | Ranking Fit judge prompt + 1/0.5/0 scale | ✅ done |
| `judges/judge_2_explanation_quality.md` | Explanation Quality judge prompt + scale | ✅ done |
| `docs/deck_outline.md` | The ≤8-slide deck structure with what goes on each | ✅ done |

### The dataset, concretely
- **10 users**, seeded sample (ids 439, 3465, 19895, 21366, 23589, 27027, 30160,
  41861, 51553, 52460) — regenerable bit-for-bit via the script.
- **Best CF model = UBCF, similarity `pearson_baseline`, k=25** — the model the
  Project 2 app ships.
- Each user's **Top-10 candidates** carry exactly the re-ranker's fields:
  `book_id, title, authors, year, average_rating, cf_score`.
- **Two contrasting requests, same for every user:**
  - *mood/tone* — "Something light, funny, and fast-paced to unwind with…"
  - *content/character* — "Something thoughtful with strong, complex character
    development…"
- **Known quirk to mention in the deck:** UBCF clips predictions to the 1–5
  scale, so all Top-10 `cf_score`s sit at the 5.0 ceiling. That's real model
  output, and it's a *feature* for this eval — the re-ranker can't just re-sort by
  confidence, it has to use the request + metadata, which is exactly what we're
  testing.

---

## 3. Rubric → to-do map

**① Evaluation Dataset & Setup (15 pts)** — ✅ built.
- Upload `eval_dataset.csv` (or `.jsonl`) as a Braintrust Dataset.
- Screenshot the dataset grid for the appendix.

**② System Instruction Prompts (20 pts)** — ✅ drafted; you finalize wording.
- Create two prompts in Braintrust from `prompts/`. Keep the user message and
  model identical; only the system instruction differs.
- Screenshot both prompts.

**③ LLM Judges (20 pts)** — ✅ drafted.
- Add the two scorers from `judges/` to the Playground/Experiment. Map
  Great/OK/Poor → 1/0.5/0 for both.
- Screenshot both judge configs.

**④ Braintrust Playground & Experiment Analysis (20 pts)** — ⬜ you run it.
- In the **Playground**: add the dataset, both prompts, both scorers. Run, eyeball
  a few rows, refine prompt/judge wording if needed.
- Create an **Experiment** for the final comparison so results are saved/shareable.
- Read individual **traces**: find ≥1 case where Prompt B clearly helped and ≥1
  where it didn't; capture cases where the two judges *disagree*.

**⑤ Slides, Communication, Evaluation critique (25 pts)** — ⬜ you build the deck.
- Follow `docs/deck_outline.md`. Include the dedicated evaluation-design slide
  (limitations, improvements, other evaluation types).

---

## 4. Step-by-step Braintrust workflow

1. **Account + project.** Sign in to Braintrust, create a project (e.g.
   `bookrec-rerank-eval`). (See the course's *Braintrust Step-by-Step Guide*.)
2. **Upload the dataset.** Datasets → New → upload `data/eval_dataset.csv`.
   Confirm 20 rows and that `candidates_json` came in intact. (Or import the
   `.jsonl` if you prefer the nested `input`/`metadata` shape.)
3. **Open a Playground.** Add the dataset.
4. **Add Prompt A.** System = `prompts/prompt_A_baseline.md` block; User =
   `prompts/user_message_template.md` block; pick your model; temperature 0.
5. **Add Prompt B** as a second prompt in the same Playground (same user message,
   same model) so they run side by side on the same rows.
6. **Add both judges** as scorers (LLM-as-a-judge), pasting from `judges/` and
   setting the Great/OK/Poor → 1/0.5/0 mapping.
7. **Run.** Iterate: if a judge is being inconsistent or a prompt misbehaves,
   tweak wording in the Playground and re-run a few rows.
8. **Create an Experiment** from the Playground for the final, saved comparison
   (this is the artifact you screenshot for the score table).
9. **Analyze traces.** Sort/filter by score, open individual rows, read the
   judge's justification. Collect your example traces.

> **Cost note:** 20 cases × 2 prompts × 2 judges = 80 judge calls + 40
> generations per full run. Each refinement run repeats that. Use a cheap model
> while iterating; do the clean final run on your chosen model.

---

## 5. Model choice (you said "not sure yet")

Default plan: **use Gemini** (`gemini-2.5-flash-lite`) for generation, since
that's what Project 2 uses — add your `GEMINI_API_KEY` in Braintrust. Same model
for generation and judging is explicitly allowed by the assignment.

If you want the judge to differ from the generator (a cleaner setup that reduces
self-preference bias), set the **judge scorers to a different model** (e.g. an
OpenAI or Anthropic model) while keeping Gemini for generation. Braintrust lets
you pick the model per scorer. Note in the deck which model did what — that's a
real limitation to discuss (see the evaluation-design slide).

**Decision needed from you:** confirm the generation model and whether the judge
uses the same or a different model.

---

## 6. Open items before you submit
- [ ] **Team number + member names** for the title slide (and confirm
      individual-vs-group matches your Project 2 submission — you can't switch).
- [ ] Confirm generation model + judge model (Section 5).
- [ ] Run the Experiment and capture all screenshots (dataset, both prompts,
      both judges, results comparison, selected traces).
- [ ] Build the deck from `docs/deck_outline.md`, export to **PDF**, ≤ 8 slides
      (+ ≤ 2 appendix), font ≥ 16.

---

## 7. How to regenerate the dataset
```bash
pip install scikit-surprise pandas
cd "project 3/scripts"
python build_eval_dataset.py
```
Knobs (users, requests, model) live at the top of `build_eval_dataset.py`.
