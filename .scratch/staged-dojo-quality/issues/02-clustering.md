# 02: Harden K-means and centroid seeding

**What to build:** Upgrade dojo exercises 0061,0075 into one editable answer with cumulative baseline, follow-up, and production checks. Focus on Empty and tied clusters; final labels consistent with centers; injected RNG; duplicate points; bounded pairwise distance memory.

**Blocked by:** None (can start immediately)

**Status:** resolved

- [x] Each question script gives a worked probe, explicit contracts for all three stages, and a live `check(..., stage=...)` cell for each stage.
- [x] Each eval script has a runnable reference and independent `verify_baseline_*`, `verify_followup_*`, and `verify_production_*` functions; later stages keep earlier behavior.
- [x] Reference evals for this ticket, judge tests, syntax checks, and `git diff --check` pass; any new production claim has an observable test or a reviewable complexity argument.

## Answer

Staged both clustering pairs. K-means assigns in bounded blocks with lowest-index ties, keeps empty centers, and returns final labels for final centers. Centroid seeding preserves distinct row indices with a zero-weight fallback and scaled weights. Targeted reference evals passed 7/7 and 9/9 checks; the judge suite passed 34 tests with 6 optional skips. Syntax and diff checks passed.
