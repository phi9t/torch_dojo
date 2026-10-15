# 13: Visualize anchor matching with `0070`

**What to build:** A learner solves anchor matching and can see assignments, collisions, and threshold effects on example boxes.

**Blocked by:** 03: Hints, new attempts, and export; 04: Explore `0059` visually

**Status:** resolved

- [x] The notebook shows the canonical contract and supports answer editing, hints, fresh-worker checking, and attempt receipts.
- [x] A threshold control updates an assignment visualization and a shape or behavior probe.
- [x] Collision and matching cases are visible with numerical evidence; exploration does not enter submitted source.
- [x] Reference passes; starter and a plausible wrong matching answer fail for the intended reasons.

## Comments

Resolved with a derived `0070` Browser Gym notebook, canonical checker bundle,
anchor-threshold visualization, tensor cards for the anchor and collision IoU
fixtures, progressive hints/reference reveal, fresh-worker qualification, and
source-bound browser receipts. Verified with focused Python tests, the
standalone browser acceptance on `DOJO_TEST_PORT=8113`, the full
`browser_gym` target, and the repository Python tests.
