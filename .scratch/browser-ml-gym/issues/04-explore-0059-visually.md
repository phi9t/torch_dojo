# 04: Explore `0059` visually

**What to build:** A learner can inspect tensor structure and vary logit scale to see its effect on cross-entropy before changing an answer.

**Blocked by:** 01: Edit and check `0059` in the browser

**Status:** resolved

- [x] A compact tensor card shows shape, dtype, named axes, and a bounded preview for explicitly supplied values.
- [x] A logit-scale control reruns the relevant probe and updates a loss plot with readable numerical evidence.
- [x] Exploration cells remain outside submitted source and cannot affect a fresh-worker check.
- [x] The plot and tensor card remain usable with keyboard navigation and legible labeling.

Resolution: Added the 0059 visual exploration cell, tensor-card helper, and browser acceptance coverage for scale updates, keyboard focus, narrow viewport visibility, and fresh-worker source isolation.
