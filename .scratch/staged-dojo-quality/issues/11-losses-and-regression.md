# 11: Deepen preference losses and regression lessons

**What to build:** Upgrade dojo exercises 0037,0038,0039,0040,0041 into one editable answer with cumulative baseline, follow-up, and production checks. Focus on Reduction and mask denominators, detached references, zero-valid cases, finite gradients, and conditioning.

**Blocked by:** None (can start immediately)

**Status:** resolved

- [x] Each question script gives a worked probe, explicit contracts for all three stages, and a live `check(..., stage=...)` cell for each stage.
- [x] Each eval script has a runnable reference and independent `verify_baseline_*`, `verify_followup_*`, and `verify_production_*` functions; later stages keep earlier behavior.
- [x] Reference evals for this ticket, judge tests, syntax checks, and `git diff --check` pass; any new production claim has an observable test or a reviewable complexity argument.

## Comments

Resolved locally: all five references passed 6/6 staged checks; 11 judge tests, syntax checks, and `git diff --check` passed. Production verifiers cover finite extreme losses and gradients, detached fixed inputs, empty batches, weighted mask denominators, validation, and least-squares conditioning.
