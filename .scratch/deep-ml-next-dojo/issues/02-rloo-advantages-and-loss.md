# 02: Practice grouped RLOO advantages and policy loss

**What to build:** A learner can compute leave-one-out response advantages and a policy-gradient loss from current log-probabilities, with optional fixed KL-shaped rewards. Reserve exercise 0090. The approved parent spec governs the common question/eval script contract.

**Blocked by:** None (can start immediately)

**Status:** resolved

- [x] The question/eval pair uses one editable top-level answer section, an original grouped-reward probe, and cumulative baseline, follow-up, and production verifiers.
- [x] Baseline computes each advantage from the *other* responses in its prompt group and returns a per-completion mean policy loss; singleton groups have zero advantage.
- [x] Follow-up handles interleaved prompt IDs, unequal group sizes, and optional KL shaping from frozen rollout and reference log-probabilities.
- [x] Production isolates gradients to current log-probabilities, returns empty advantages and graph-connected zero loss for an empty batch, validates inputs, and mutates nothing.
- [x] The reference passes direct evaluation; saved candidate stage selection works; global reward normalization and self-inclusion in the baseline both fail checks. Add the exercise to the dojo index with a source link and original wording.

## Comments

- Integrated `2c1fcc3` as exercise 0090. Direct reference evaluation passed 6/6 checks; saved candidate checks passed baseline 1/1, follow-up 3/3, and production 6/6. The baseline check rejected both global reward normalization and self-including group baselines. The README index links the original question and eval scripts and cites the motivating Deep-ML topic.
