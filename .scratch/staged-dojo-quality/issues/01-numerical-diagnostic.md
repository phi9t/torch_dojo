# 01: Harden stable CE and classifier training

**What to build:** Upgrade dojo exercises 0059,0060 into one editable answer with cumulative baseline, follow-up, and production checks. Focus on Stable logits and manual gradients; ignored targets and all-ignored policy; valid-count weighted accumulation; deterministic decreasing-loss example.

**Blocked by:** None (can start immediately)

**Status:** resolved

- [x] Each question script gives a worked probe, explicit contracts for all three stages, and a live `check(..., stage=...)` cell for each stage.
- [x] Each eval script has a runnable reference and independent `verify_baseline_*`, `verify_followup_*`, and `verify_production_*` functions; later stages keep earlier behavior.
- [x] Reference evals for this ticket, judge tests, syntax checks, and `git diff --check` pass; any new production claim has an observable test or a reviewable complexity argument.
