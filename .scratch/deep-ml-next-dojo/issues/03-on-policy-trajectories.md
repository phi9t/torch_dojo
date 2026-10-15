# 03: Practice on-policy MDP trajectory sampling

**What to build:** A learner can sample ordered trajectories from a stochastic policy and transition model, then report visit frequencies, state-action counts, and mean discounted return. Reserve exercise 0091. The approved parent spec governs the common question/eval script contract.

**Blocked by:** None (can start immediately)

**Status:** resolved

- [x] The question/eval pair uses top-level cells, one editable answer section, a prediction probe, and cumulative baseline, follow-up, and production verifiers.
- [x] Baseline records a deterministic hand-computable episode and counts the initial state and each reached successor exactly once.
- [x] Follow-up uses an injected NumPy generator to sample both actions and transitions, stops at terminal states or the horizon, and averages discounted return by episode.
- [x] Production validates finite probability rows and shapes, preserves inputs, handles a terminal starting state, and conserves counts between trajectory records and aggregate outputs.
- [x] The reference passes direct evaluation; saved candidate stage selection works; a greedy sampler or one that acts after termination fails checks. Add the exercise to the dojo index with a source link and original wording.

## Resolution

Merged `0091_on_policy_trajectories.py` and its eval with the #91 index row. Direct reference evaluation passed 7/7 verifiers. Saved candidate baseline selection ran exactly 2 checks and rejected the blank answer as expected. The seeded stochastic trace and terminal-start checks distinguish greedy sampling and acting after termination.
