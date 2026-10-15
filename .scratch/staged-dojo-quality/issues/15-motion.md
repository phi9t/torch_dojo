# 15: Deepen trajectory rollout, metrics, modes, and collision

**What to build:** Upgrade dojo exercises 0055,0056,0057,0058 into one editable answer with cumulative baseline, follow-up, and production checks. Focus on Irregular time, non-prefix masks, all-invalid policies, shared-mode selection, and collision boundary semantics.

**Blocked by:** None (can start immediately)

**Status:** resolved

- [x] Each question script gives a worked probe, explicit contracts for all three stages, and a live `check(..., stage=...)` cell for each stage.
- [x] Each eval script has a runnable reference and independent `verify_baseline_*`, `verify_followup_*`, and `verify_production_*` functions; later stages keep earlier behavior.
- [x] Reference evals for this ticket, judge tests, syntax checks, and `git diff --check` pass; any new production claim has an observable test or a reviewable complexity argument.

## Comments

Implemented all four motion pairs with cumulative stages. The reference evals (15 checks), judge tests (11), syntax checks, and diff whitespace check passed.
