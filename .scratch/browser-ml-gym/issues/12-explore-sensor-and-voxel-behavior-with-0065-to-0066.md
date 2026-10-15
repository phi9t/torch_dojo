# 12: Explore sensor and voxel behavior with `0065`–`0066`

**What to build:** A learner inspects sensor transformations and voxel contributions while solving both exercises in the browser.

**Blocked by:** 03: Hints, new attempts, and export; 04: Explore `0059` visually

**Status:** resolved

- [x] Each notebook preserves the canonical contract and provides editable answers, hints, isolated checks, and attempt receipts.
- [x] Visual probes reveal transformed coordinates or voxel contributions, and tensor cards make shape and dtype clear.
- [x] A relevant parameter control updates its probe without automatically launching long work or changing submitted source.
- [x] Each reference passes, and each starter and selected plausible wrong answer fail for the intended reasons.

## Comments

Follow-up browser acceptance added for `0065` and `0066`: each generated notebook opens through the lab UI, runs its visual probe, updates a probe-only control, keeps the submitted answer source unchanged, and verifies the canonical reference in a fresh worker. The 0065 probe now uses the canonical z-rotation worked example, and the 0066 voxel SVG draws within its viewBox.
