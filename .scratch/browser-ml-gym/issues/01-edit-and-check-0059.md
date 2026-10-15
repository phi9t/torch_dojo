# 01: Edit and check `0059` in the browser

**What to build:** A learner can open the stable cross-entropy exercise as a browser notebook, edit its answer, run Python, and check the answer against the canonical exercise in a fresh worker.

**Blocked by:** None (can start immediately)

**Status:** resolved

- [x] The browser opens an editable notebook derived from the tagged question script, with the exercise contract and examples visible.
- [x] The learner can run a cell and check only designated solution cells; scratch and probe cells cannot alter the submitted source.
- [x] A fresh worker checks the exact submitted source against the canonical verifier, and shows a clear pass or failure with measured evidence.
- [x] The reference answer passes, while the starter and a plausible numerically unstable answer fail for the intended reasons.

## Comments

Implemented against the current three-stage `0059` question and all seven canonical verifiers. The learner selects baseline, follow-up, or production, with cumulative checks as in the native workflow. Browser acceptance covers running a cell, starter failure, reference pass, unstable softmax failure, wrong batch-gradient failure, traceback/timing evidence, and scratch state isolation. The browser worker terminates after each check.
