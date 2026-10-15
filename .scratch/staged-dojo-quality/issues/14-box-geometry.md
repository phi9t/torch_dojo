# 14: Deepen box IoU, sampling, and membership

**What to build:** Upgrade dojo exercises 0062,0063,0064 into one editable answer with cumulative baseline, follow-up, and production checks. Focus on Continuous-coordinate boundaries, ties, degenerate boxes, inverse-transform invariants, and batched memory.

**Blocked by:** None (can start immediately)

**Status:** resolved

- [x] Each question script gives a worked probe, explicit contracts for all three stages, and a live `check(..., stage=...)` cell for each stage.
- [x] Each eval script has a runnable reference and independent `verify_baseline_*`, `verify_followup_*`, and `verify_production_*` functions; later stages keep earlier behavior.
- [x] Reference evals for this ticket, judge tests, syntax checks, and `git diff --check` pass; any new production claim has an observable test or a reviewable complexity argument.

## Comments

Resolved with cumulative checks for continuous xyxy IoU, seeded yaw-box sampling, and point membership. The two pair-grid references use 256 by 256 pair blocks, so the mandatory output uses O(NM) space while temporary pair arrays remain O(block squared). Targeted reference evals pass 6/6, 6/6, and 7/7; repository tests pass 34 tests with 6 optional skips.
