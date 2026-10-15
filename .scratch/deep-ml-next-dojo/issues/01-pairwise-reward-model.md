# 01: Practice pairwise reward-model training

**What to build:** A learner can implement a scalar reward-model pair loss and strict preference accuracy, then strengthen it through margin and numerical-stability follow-ups. Reserve exercise 0089. The approved parent spec governs the common question/eval script contract.

**Blocked by:** None (can start immediately)

**Status:** resolved

- [x] The question script uses top-level Emacs/Jupyter cells, one editable answer section, an original hand-calculation probe, and cumulative baseline, follow-up, and production checks; the companion eval script holds the reference and native verifiers.
- [x] Baseline returns differentiable mean pairwise logistic loss and strict chosen-over-rejected accuracy for finite equal-length score vectors.
- [x] Follow-up applies a nonnegative margin to loss while accuracy still uses unshifted scores; a tie counts as incorrect.
- [x] Production rejects malformed inputs, handles extreme wrong-way gaps with finite loss and gradients, and leaves inputs unchanged.
- [x] The reference passes direct evaluation; saved candidate stage selection works; a swapped or constant-loss candidate fails a decisive check. Add the exercise to the dojo index with a source link and original wording.

## Resolution

Merged `0089_pairwise_reward.py` and its eval, preserving the #89 and #90 index entries. The shared Python environment ran the direct reference eval: 5/5 verifiers passed. The saved blank candidate reached the production checks and failed as expected; the baseline swapped-score verifier distinguishes reversed or constant losses.
