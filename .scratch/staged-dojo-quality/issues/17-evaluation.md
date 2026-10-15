# 17: Deepen calibration, matching, bootstrap, and AP

**What to build:** Upgrade dojo exercises 0069,0070,0073,0074 into one editable answer with cumulative baseline, follow-up, and production checks. Focus on Denominators, absent classes, deterministic ties, paired resampling, one-to-one claims, and metric definitions.

**Blocked by:** 14: box-geometry

**Status:** resolved

- [x] Each question script gives a worked probe, explicit contracts for all three stages, and a live `check(..., stage=...)` cell for each stage.
- [x] Each eval script has a runnable reference and independent `verify_baseline_*`, `verify_followup_*`, and `verify_production_*` functions; later stages keep earlier behavior.
- [x] Reference evals for this ticket, judge tests, syntax checks, and `git diff --check` pass; any new production claim has an observable test or a reviewable complexity argument.

## Comments

Resolved with staged calibration metrics, deterministic anchor matching, trip-paired percentile bootstrap, and all-point interpolated detection AP. Bootstrap stores one trip-index draw and one scalar per replicate, using O(T+iterations) auxiliary memory. Reference checks for each pair pass 6/6; repository judge tests pass 34 with 6 optional skips; syntax and diff checks pass.
