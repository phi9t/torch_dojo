# %%
# 0073 · Paired trip bootstrap
# Goal: Estimate how much a candidate model improves on a baseline, with a
# percentile bootstrap confidence interval that resamples whole trips and
# keeps each trip's two losses paired.
# Authored practice variant; see practice_bank/drills.tsv for related drills.
# Revise the same answer cell as you advance through all three stages.

# %%
import numpy as np

# %%
# Baseline — Each entry is one trip-level loss. Improvement is baseline
# minus candidate: positive means the candidate has lower loss.
probe_base = np.array([2., 4., 6.])
probe_new = np.array([1., 3., 5.])
print("trip improvements:", probe_base - probe_new)
# Return `(mean improvement, (lo, hi))`, where `lo` and `hi` are the 2.5th and
# 97.5th percentiles of the replicate means. For each of `iterations`
# bootstrap replicates, draw $T$ trip indices with replacement using the
# injected NumPy `Generator`. $T \ge 2$; `iterations` $\ge 1$.
#
# Predict: The point estimate and the interval for three trip improvements of 1.

# %% [drill]
# Paired or independent resampling?
# Every trip improves by exactly 1. Predict both 95% intervals.
import numpy as np
baseline = np.array([1., 5., 9.])
candidate = np.array([0., 4., 8.])
rng = np.random.default_rng(0)
paired, independent = [], []
for _ in range(200):
    i = rng.integers(0, 3, size=3)
    j = rng.integers(0, 3, size=3)
    paired.append((baseline[i] - candidate[i]).mean())
    independent.append(baseline[i].mean() - candidate[j].mean())
print(np.percentile(paired, [2.5, 97.5]), np.percentile(independent, [2.5, 97.5]).round(3))

# %% [drill]
# Percentile of five values
# Predict the 2.5th and 97.5th percentiles. Are they values from the array?
import numpy as np
print(np.percentile(np.array([0., 1., 2., 3., 4.]), [2.5, 97.5]))

# %% [drill]
# Trips or frames?
# Trip A has one frame that improves by 10; trip B has nine frames that
# improve by 0. Predict the trip mean and the frame mean.
import numpy as np
trips = [np.array([10.]), np.zeros(9)]
print(np.mean([trip.mean() for trip in trips]), np.concatenate(trips).mean())

# %% [answer]
def paired_trip_bootstrap(baseline: np.ndarray, candidate: np.ndarray,
                          iterations: int, rng: np.random.Generator
                          ) -> tuple[float, tuple[float, float]]:
    pass

# %%
from torch_judge import check
check("0073_paired_trip_bootstrap_eval.py", stage="baseline")

# %%
# Follow-up — Resample paired trip DIFFERENCES, using the same sampled
# indices for both models.
# Watch out: Do not sample frames, or each model separately. Independent
# draws break the pairing and add the spread between trips to the interval.
#
# The percentile CI uses `np.percentile`'s default linear interpolation. A
# seeded `Generator` should replay exactly. The point estimate comes from the
# original trips, not from the replicates.
check("0073_paired_trip_bootstrap_eval.py", stage="followup")

# %%
# Production — Reject mismatched, non-1D or non-finite inputs, fewer than two
# trips, and nonpositive or noninteger `iterations`. Do not mutate inputs.
# Avoid an `[iterations,T]` index matrix: process replicates one at a time
# and store only $O(T + \text{iterations})$ values.
# Why: The trip is the unit that was sampled, so the trip mean weights every
# trip once. It does not weight by frame count or exposure.
check("0073_paired_trip_bootstrap_eval.py", stage="production")
