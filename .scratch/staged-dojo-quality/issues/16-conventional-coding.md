# 16: Deepen RLE, intervals, and grid components

**What to build:** Upgrade dojo exercises 0067,0068,0071 into one editable answer with cumulative baseline, follow-up, and production checks. Focus on Large counts without expansion, half-open endpoints, malformed input, iterative traversal, and streaming follow-ups.

**Blocked by:** None (can start immediately)

**Status:** resolved

- [x] Each question script gives a worked probe, explicit contracts for all three stages, and a live `check(..., stage=...)` cell for each stage.
- [x] Each eval script has a runnable reference and independent `verify_baseline_*`, `verify_followup_*`, and `verify_production_*` functions; later stages keep earlier behavior.
- [x] Reference evals for this ticket, judge tests, syntax checks, and `git diff --check` pass; any new production claim has an observable test or a reviewable complexity argument.

## Comments

Resolved with staged RLE, interval union, and grid component scripts. Targeted reference evals pass 18/18 verifiers; the judge suite passes 11/11 tests. Syntax checks and `git diff --check` pass. The production cells state the complexity contracts, with huge-count, large-chain, order-invariance, immutability, and generated cases as observable checks.
