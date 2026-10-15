# 19: Verify staged dojo coverage and review the migration

**What to build:** Verify that every one of the 81 dojo pairs has an explicit baseline, follow-up, and production contract or a documented exercise-specific exception; review the migration against the staged-quality spec and close the remaining quality gaps.

**Blocked by:** 01–18: all exercise migration tickets.

**Status:** resolved

- [x] Stage coverage is checked for all 81 pairs, including question cells and eval verifiers.
- [x] `make eval`, repository tests, syntax checks, and diff checks pass; performance or memory claims receive targeted review.
- [x] Update the README, audit, and plan to reflect the actual completed scope, then land locally.

## Comments

Resolved after a structural audit of all 81 pairs: exactly one tagged answer cell, a question-side `check` call for each stage, and at least one eval verifier per stage. Counts are 128 baseline, 146 follow-up, and 171 production verifiers. All 81 reference eval scripts passed all 445 checks; repository tests passed 34 tests with 6 optional skips; 171 Python sources compiled; diff check passed. Reviewed bounded-workspace claims in blocked IoU and K-means assignment, local/tiled attention, sparse voxel pooling, and chunked CSV. Corrected oriented-box sampling so malformed yaw inputs consistently raise the documented `ValueError`.
