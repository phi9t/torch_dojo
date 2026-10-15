# 07: Deepen long-sequence and cache attention

**What to build:** Upgrade dojo exercises 0012,0013,0014,0023,0025 into one editable answer with cumulative baseline, follow-up, and production checks. Focus on Cache equivalence, cross-attention lengths, causal correctness, block boundaries, precision, and memory growth.

**Blocked by:** 06: attention-family

**Status:** resolved

- [x] Each question script gives a worked probe, explicit contracts for all three stages, and a live `check(..., stage=...)` cell for each stage.
- [x] Each eval script has a runnable reference and independent `verify_baseline_*`, `verify_followup_*`, and `verify_production_*` functions; later stages keep earlier behavior.
- [x] Reference evals for this ticket, judge tests, syntax checks, and `git diff --check` pass; any new production claim has an observable test or a reviewable complexity argument.

## Comments

Resolved locally: references and staged verifiers now cover unequal lengths,
cache chunks, causal prefixes, block tails, numerical range, validation and
autograd. All five reference evals pass; `tests/test_judge.py` passes 11/11.
