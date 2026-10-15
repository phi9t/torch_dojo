# %%
# 0061 · K-Means: assign, update, and scale the same kernel
# Goal: Implement Lloyd's K-means in NumPy: assign each point to its nearest
# center, move each center to the mean of its points, and keep the same kernel
# correct as clusters empty out and inputs grow.
# Authored practice based on the reported K-means topic. The exact contracts
# below are authored extensions, not imported task constraints.

# %%
import numpy as np

# %%
# Baseline — Lloyd updates from the given initial centers.
# Predict: Start with `X=[0,1,9,10]` and `centers=[0,10]`. What are the first
# assignment, the updated centers, and the inertia?
X_probe = np.array([[0.], [1.], [9.], [10.]])
print("points:", X_probe.ravel(), "initial centers:", [0., 10.])
# Predict: What are `centers_probe[labels_probe].shape` and the gathered values?
centers_probe = np.array([[0.], [10.]])
labels_probe = np.array([0, 0, 1, 1])
print("assigned centers:", centers_probe[labels_probe])
# Implement the first update in your answer, then check the sums against
# `[1,19]`, counts against `[2,2]`, and means against `[0.5,9.5]`.
#
# Contract: finite numeric `X` $[N,D]$, `initial_centers` $[K,D]$, $N,K,D \ge 1$.
# Run at most `max_iter` Lloyd updates. Return `(centers [K,D], labels [N],
# inertia)`. Inertia is the sum of squared distances to the returned centers.

# %% [drill]
# Repeated destinations
# Three points all belong to cluster 0. Predict both sums.
import numpy as np
sums = np.zeros(1)
sums[[0, 0, 0]] += np.array([2., 3., 5.])
print(sums)
sums = np.zeros(1)
np.add.at(sums, [0, 0, 0], np.array([2., 3., 5.]))
print(sums)

# %% [drill]
# Counting a cluster nobody joined
# There are three centers, but no point picked center 2. Predict both counts
# and the means.
import numpy as np
labels = np.array([0, 0, 1])
print(np.bincount(labels), np.bincount(labels, minlength=3))
with np.errstate(invalid="ignore"):
    print(np.array([1., 9., 0.]) / np.bincount(labels, minlength=3))

# %% [drill]
# Are the labels still current?
# One assign, one update, then assign again. Predict both label vectors.
import numpy as np
X = np.array([0., 1., 9., 10.])
centers = np.array([0., 1.])
labels = np.abs(X[:, None] - centers[None, :]).argmin(axis=1)
centers = np.array([X[labels == k].mean() for k in range(2)])
fresh = np.abs(X[:, None] - centers[None, :]).argmin(axis=1)
print(labels, centers.round(3), fresh)

# %% [answer]
# Improve this one definition in place as you advance through the stages.
def kmeans(X: np.ndarray, initial_centers: np.ndarray, max_iter: int = 100,
           *, block_size: int = 1024) -> tuple[np.ndarray, np.ndarray, float]:
    pass

# %%
from torch_judge import check
check("0061_kmeans_eval.py", stage="baseline")

# %%
# Gather/scatter trap: predict this update before executing the cell.
# Gathering reads assigned centers; scattering sends points back to clusters.
buffered_sums = np.zeros_like(centers_probe)
buffered_sums[labels_probe] += X_probe
print("buffered sums:", buffered_sums.ravel())
# Watch out: Buffered `sums[labels] += X` writes each repeated destination
# once, so later points overwrite earlier ones instead of adding up.
# Repeated destinations need accumulation. Implement a sum buffer using
# `np.add.at` in your answer. Accumulate counts too; preserve empty centers.
# Add `[2,3,5]` into one cluster: the sum must be 10, not one selected point.
# Hints and a worked accumulating update live in the separate eval file.

# %%
# Follow-up — Empty clusters keep their previous centers. Assignment ties
# choose the lowest center index. Even with `max_iter=0`, return labels and
# inertia for the initial centers.
# Watch out: After the final update, reassign points to the returned centers.
# Labels from the previous iteration can be stale.
# Predict: With duplicated points and three centers, which center stays empty?
check("0061_kmeans_eval.py", stage="followup")

# %%
# Production — Validate shapes, finite values, limits, and numeric types.
# Do not mutate `X` or `initial_centers`. `block_size` limits the assignment
# workspace: a `[block_size,D]` difference buffer plus $O(N+KD)$ state is
# enough; avoid an `[N,K,D]` tensor. A sequential loop over $K$ is fine.
# Compare `block_size=1` with the default. The judge checks answers across
# block sizes, not peak allocation. Review the assignment code's live arrays
# to verify the workspace claim by hand.
# Explain the time cost $O(TNKD)$ for $T$ = `max_iter`, and why the norm
# identity $\lVert x-c\rVert^2 = \lVert x\rVert^2 - 2x \cdot c + \lVert c\rVert^2$ may lose
# precision when coordinates share a large common offset.
check("0061_kmeans_eval.py", stage="production")

# %%
# Transfer: `embedding[token_ids]` gathers vectors. Repeated tokens send their
# gradient contributions to the SAME embedding row, so backward accumulates.
# A point-cloud neighborhood gathers points by neighbor IDs; overlapping
# neighborhoods likewise accumulate contributions at shared source points.
# Compare with cross-entropy: repeated class IDs in DIFFERENT batch rows
# address different (row,class) cells. A shared class ID alone is no collision.
# Next: 0066_voxel_pool.py for repeated spatial destinations, then geometry.
