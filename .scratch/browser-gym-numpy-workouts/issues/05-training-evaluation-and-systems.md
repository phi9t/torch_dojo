# 05: Training, evaluation and systems: 0030, 0050, 0075, 0100, 0094, 0080

**What to build:** The last six NumPy exercises join the gym: the cosine schedule with warmup, reverse-mode differentiation through a scalar graph, K-means++ landmarks and expert capacity admission (training), binary calibration (evaluation), and coaster scoring (systems).

**Blocked by:** 01

**Status:** done

- [x] Each has a metadata entry with its topic first, a typed brief, and 2–3 drills (warmup boundary steps, accumulating gradients on a shared node, seeded D² sampling, tie-breaking when an expert is full, fitting on logits vs probabilities, integer vs float scoring).
- [x] Each is in the qualification e2e table and the smoke spec.
- [x] Native evals and question scripts behave as before; pytest passes.

## Comments

- Drills: 0030 (warmup start, progress denominator, past the last step), 0050 (assign vs accumulate, walking backward), 0075 (distance vs squared distance, seeded draws, nearest vs latest center), 0100 (tie order under a reversed sort, rank pool size), 0094 (sigmoid on logits, slope penalty on separable data, tied scores), 0080 (true vs floor division, the Steel cap).
- Native verifier counts unchanged: 0030 6/6, 0050 5/5, 0075 9/9, 0100 7/7, 0094 6/6, 0080 4/4.
- Review fixes: 0075 drill 3's reasoning names the right point (point 1, next to center 0); the 0050 and 0030 Goals end before the attribution and workflow notes.
