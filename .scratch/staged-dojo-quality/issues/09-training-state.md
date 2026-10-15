# 09: Deepen optimizer, scheduler, and accumulation lessons

**What to build:** Upgrade dojo exercises 0029,0030,0031 into one editable answer with cumulative baseline, follow-up, and production checks. Focus on Optimizer state, uneven microbatches, step boundaries, reproducibility, and equivalence to a full-batch update.

**Blocked by:** None (can start immediately)

**Status:** resolved

- [x] Each question script gives a worked probe, explicit contracts for all three stages, and a live `check(..., stage=...)` cell for each stage.
- [x] Each eval script has a runnable reference and independent `verify_baseline_*`, `verify_followup_*`, and `verify_production_*` functions; later stages keep earlier behavior.
- [x] Reference evals for this ticket, judge tests, syntax checks, and `git diff --check` pass; any new production claim has an observable test or a reviewable complexity argument.

## Answer

Exercises 0029–0031 now use one editable answer cell with cumulative baseline, follow-up, and production checks. Adam tracks per-parameter steps when gradients are absent; cosine scheduling defines zero/full warmup and validates its domain; accumulation weights uneven microbatches by sample count and supports one-pass streams. All targeted reference and judge checks passed.
