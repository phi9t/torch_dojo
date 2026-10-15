# 01: Tracer bullet: 0068 interval union in the browser gym

**What to build:** 0068 becomes a browser workout end to end, in the 0059 form, and the pattern every later batch copies is settled: metadata entry, typed brief, drills, a table-driven qualification e2e that needs no explore panel, and the smoke spec.

**Blocked by:** None (can start immediately)

**Status:** done

- [x] 0068 has a `QUALIFIED_DOJO_EXERCISES` entry (topic "arrays" first, three progressive hints) and is in `DEFAULT_CATALOG_IDS`; it shows under Arrays and indexing on the home page and in the ⌘K switcher.
- [x] Its brief opens with a Goal and marks traps and explanations where they fit; it has 2–3 drills with `# Drill <n>:` reasoning in the eval script.
- [x] A table-driven qualification e2e (new, for workouts without explore panels) shows the reference passing at production, the starter failing at baseline, and one mutant failing for its intended reason, in a fresh worker.
- [x] The converted-workout smoke includes 0068.
- [x] The eval's imports load in the checker worker; native eval verifier count and question-script behaviour are unchanged; pytest passes.

## Comments

- The pattern for later batches: a `QUALIFIED_DOJO_EXERCISES` entry (topic first in `concepts`, three progressive hints) and the id appended to `DEFAULT_CATALOG_IDS`; stage headings converted from `Baseline:` to `Baseline —` so the brief parser scopes them; drills and `# Drill <n>:` reasoning as in 0059; one row in `numpy_workouts.spec.ts` (`id`, a one-line `swap(from, to)` mutation of the reference, and the verifier it must fail); the id in `converted.spec.ts`.
- `numpy_workouts.spec.ts` takes the reference from the built catalog's `solution`, so it covers whatever imports and helpers the eval's reference needs.
- `home.spec.ts` now reads the workout list from the catalog instead of a hard-coded list.
- 0068's mutant replaces `end = max(end, hi)` with `end = hi` and fails `verify_baseline_empty_and_nested`.
