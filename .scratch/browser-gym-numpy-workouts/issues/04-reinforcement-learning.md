# 04: Reinforcement learning: 0082, 0083, 0091

**What to build:** The reinforcement learning exercises join the gym under a new topic: policy evaluation, Q-learning from a replay tape, and on-policy trajectories.

**Blocked by:** 01

**Status:** done

- [x] Each has a metadata entry (topic "reinforcement learning"), a typed brief with the Bellman updates as math, and 2–3 drills (terminal-state bootstrapping, the order of updates within a sweep, discounting, sampling with a seeded generator).
- [x] Each is in the qualification e2e table and the smoke spec; the `numbers` import loads in the checker worker.
- [x] Native evals and question scripts behave as before; pytest passes.

## Comments

- Drills: 0082 (bootstrapping from terminal states, synchronous vs in-place sweeps, how far discounting reaches), 0083 (ignoring `done`, frozen vs updated table, an aliased table), 0091 (seeded draw order, the undiscounted first step, counting the start state).
- Native verifier counts unchanged: 7/7 each. 0091 and 0100 exposed that the browser checker never received helpers defined in the verifiers cell; fixed in the bundle (commit "Send the verifiers' helpers to the browser checker"), pinned by a pytest on 0091's bundle.
