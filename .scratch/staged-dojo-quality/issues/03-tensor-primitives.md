# 03: Deepen activation, softmax, linear, and LayerNorm lessons

**What to build:** Upgrade dojo exercises 0001,0002,0003,0004 into one editable answer with cumulative baseline, follow-up, and production checks. Focus on Scalar and batched shapes; stable reductions; dtype/device and autograd; noncontiguous inputs; input mutation rules.

**Blocked by:** None (can start immediately)

**Status:** resolved

- [x] Each question script gives a worked probe, explicit contracts for all three stages, and a live `check(..., stage=...)` cell for each stage.
- [x] Each eval script has a runnable reference and independent `verify_baseline_*`, `verify_followup_*`, and `verify_production_*` functions; later stages keep earlier behavior.
- [x] Reference evals for this ticket, judge tests, syntax checks, and `git diff --check` pass; any new production claim has an observable test or a reviewable complexity argument.

## Answer

Staged ReLU, softmax, SimpleLinear, and LayerNorm as interactive scripts with one answer and three cumulative checks each. Reference verifiers cover known values, noncontiguous and higher-rank inputs, gradients, dtype/device behavior, input immutability, and the specified validation contracts. Targeted reference evals passed 5/5, 5/5, 6/6, and 6/6 checks. The judge suite passed 11 tests. Syntax and diff checks passed.
