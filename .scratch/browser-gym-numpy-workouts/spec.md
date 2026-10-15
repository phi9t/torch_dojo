# Browser gym: bring the NumPy exercises in

Status: done

Sources: the browser gym polish (`.scratch/browser-gym-polish/`, done), the user's request of 2026-10-09 to migrate the other examples to the 0059 form, and their choice to start with the exercises that can run in the browser. Terms follow `GLOSSARY.md` (workout, drill, rep, split, answer cell).

## Problem Statement

The browser gym has 12 dojo workouts plus the MLP lab, and they now read the way 0059 does: a typed brief, drills, the Bench, timed reps. The repository has 101 more exercises, and I can only practise them natively.

83 of them need PyTorch, which Pyodide cannot run, so they stay native for now. The other 18 use only NumPy and the standard library, and their scripts already follow the tagged-cell contract (ADR 0001) with three cumulative stages (ADR 0002). Nothing stops them from being browser workouts except the qualification work and the brief and drill authoring.

## Solution

Qualify the 18 NumPy-only exercises as browser workouts, each in the 0059 form:

| Topic on the home page | Workouts |
| --- | --- |
| Arrays and indexing | 0067 run-length random access, 0068 interval union, 0071 grid components, 0076 max-pool coordinates, 0077 sparse COO |
| Parsing | 0035 BPE-style tokenizer, 0072 CSV records, 0078 CSV consumer, 0079 two-operator calculator |
| Reinforcement learning | 0082 policy evaluation, 0083 Q-learning replay, 0091 on-policy trajectories |
| Training | 0030 cosine schedule with warmup, 0050 reverse-mode scalar graph, 0075 K-means++ landmarks, 0100 expert capacity admission |
| Evaluation | 0094 binary calibration |
| Systems | 0080 coaster scoring |

Each one:

- appears on the home page under its topic, in the ⌘K switcher, and at `?exercise=<id>`;
- has a typed brief (Goal, then Predict / Watch out / Why where they fit, `$…$` math), scoped by its existing stage headings;
- has two or three drills aimed at its most common misconception, with reasoning in its eval script;
- is checked in a fresh Pyodide worker: the reference passes every stage, the starter fails, and one plausible wrong answer fails for the intended reason;
- keeps its native `make test` / `make eval` behaviour unchanged.

## User Stories

1. As a learner, I want the NumPy exercises on the home page, grouped by topic, so that I can drill them in the browser like 0059.
2. As a learner, I want each new workout's brief to state its goal first and mark its traps, so that it reads like the converted ones.
3. As a learner, I want drills on each new workout, so that I can test my intuition before checking.
4. As a learner, I want checks of the new workouts to run in a fresh worker with the same receipts, grid and splits, so that nothing about reps changes.
5. As a learner, I want Next up to include the new workouts, so that they come up in my rotation.
6. As a workout author, I want qualifying an exercise to be a metadata entry plus authored comments, so that adding the next one is cheap.

## Implementation Decisions

- **Qualification stays table-driven.** Each workout gets an entry in `QUALIFIED_DOJO_EXERCISES` (title, stages, concepts with the topic first, three progressive hints, `allowed_helpers`) and its id in `DEFAULT_CATALOG_IDS`. No per-workout build code.
- **Explore panels are optional.** A workout without a `browser_gym/lessons/<id>.py` has no Explore panels; the Explore tab shows zero. Lessons can come later, one at a time.
- **Briefs and drills follow the existing markers** (`# Goal:` opt-in, `# %% [drill]` cells, `# Drill <n>:` reasoning in the eval script). Only comments and drill cells change in the scripts.
- **Browser checks reuse the existing bundle path.** If an eval needs something the checker worker doesn't provide (a stdlib module Pyodide lacks, a Hypothesis profile too slow for the browser), fix it in the checker runtime or the eval's settings for the browser, not by forking the verifiers. Record any such change in the ticket.
- **Qualification e2e is table-driven too.** One spec lists, per workout: the candidate function, the reference (from the eval), the starter (from the question), and one mutant with its expected failure text.
- **New topics** (arrays, parsing, reinforcement learning, systems) are new first concepts. The home page groups by first concept, so no UI change is needed.

## Testing Decisions

Same rules as the polish spec: test outside-visible behaviour, never CSS classes or internals.

- pytest: the catalog builds every new workout with a goal-first brief and 2–3 drills; every catalog entry has a topic; drill cells never reach the answer cell.
- e2e (Playwright): for each new workout, the reference passes at production, the starter fails at baseline, and the mutant fails for its intended reason; the converted-workout smoke opens it and runs a drill.
- Native: each eval script passes with the same verifier count as before; each question script runs to its first unsolved assertion as before.

## Out of Scope

- The 83 PyTorch exercises (they cannot run in Pyodide). Converting their briefs and drills for the native workflow is a possible later effort.
- New explore lessons for the 18 (optional follow-ups).
- Changes to verifier semantics.

## Further Notes

- `0091` and `0083` import `numbers`, which Pyodide's stdlib includes; confirm in the tracer that every eval's imports load in the checker worker.
- The full e2e suite takes about 13 minutes today and will grow by roughly a minute per batch of qualification cases.
