# 06: Deepen softmax, causal, grouped, and window attention

**What to build:** Upgrade dojo exercises 0005,0009,0010,0011 into one editable answer with cumulative baseline, follow-up, and production checks. Focus on Mask semantics, all-masked rows, head sharing, window boundaries, batch isolation, and backward parity with explicit shapes.

**Blocked by:** None (can start immediately)

**Status:** resolved

- [x] Each question script gives a worked probe, explicit contracts for all three stages, and a live `check(..., stage=...)` cell for each stage.
- [x] Each eval script has a runnable reference and independent `verify_baseline_*`, `verify_followup_*`, and `verify_production_*` functions; later stages keep earlier behavior.
- [x] Reference evals for this ticket, judge tests, syntax checks, and `git diff --check` pass; any new production claim has an observable test or a reviewable complexity argument.

## Comments

Resolved with staged contracts and 26 reference checks across the four exercises.
Sliding-window reference processes bounded key slices rather than forming a
dense score matrix; its per-query loop is deliberate. Verified with four
reference eval runs, 11 judge tests, syntax compilation, and diff check.
