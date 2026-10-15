# 14: Explore grouped evaluation with `0073`–`0074`

**What to build:** A learner solves grouped bootstrap and detection average precision exercises while visualizing uncertainty and precision-recall behavior.

**Blocked by:** 03: Hints, new attempts, and export; 04: Explore `0059` visually

**Status:** resolved

- [x] Both notebooks preserve their canonical metric conventions and provide editable answers, hints, isolated checks, and attempt receipts.
- [x] Bootstrap and precision-recall plots update from meaningful bounded probes, with shape or behavior evidence shown nearby.
- [x] Controls affect only relevant probes and cannot change the source submitted for checking.
- [x] Each reference passes, and each starter and selected plausible wrong answer fail for the intended reasons.

## Comments

Resolved with generated browser lessons for `0073` and `0074`, table-driven
catalog metadata, canonical-reference extraction that omits verifier-only
imports, and qualification coverage for reference, starter, wrong-sign
bootstrap, and wrong-recall-denominator AP variants. Verified with focused
Python tests, `make -C browser_gym build`, `DOJO_TEST_PORT=8114 make -C
browser_gym test`, and the repository Python suite.
