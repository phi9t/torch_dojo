# 18: Deepen new pooling, sparse, CSV consumer, parser, coaster, and module lessons

**What to build:** Upgrade dojo exercises 0076,0077,0078,0079,0080,0081 into one editable answer with cumulative baseline, follow-up, and production checks. Focus on Repeated-index accumulation, error records, parser grammar, parameter state, invalid inputs, and C++-to-Python adaptation limits.

**Blocked by:** None (can start immediately)

**Status:** resolved

- [x] Each question script gives a worked probe, explicit contracts for all three stages, and a live `check(..., stage=...)` cell for each stage.
- [x] Each eval script has a runnable reference and independent `verify_baseline_*`, `verify_followup_*`, and `verify_production_*` functions; later stages keep earlier behavior.
- [x] Reference evals for this ticket, judge tests, syntax checks, and `git diff --check` pass; any new production claim has an observable test or a reviewable complexity argument.

## Comments

Resolved with all six staged pairs. Targeted reference checks pass (7, 6, 6, 6, 4, and 6 checks respectively); judge suite passes 34 tests with 6 optional skips. Syntax and diff checks pass. The pooling lesson explains repeated backward accumulation without adding a second answer target; COO directly verifies repeated-index accumulation in both products.
