# %%
# 0066 · Voxelize points without building a dense city of empty cells
# Goal: Implement `voxel_pool(points, features, origin, voxel_size, shape)` in
# NumPy: assign each point to a voxel and return only the occupied voxels, with
# point counts and mean features. Keep one answer and strengthen it through
# three stages.
#
# Practice scene: the parking lot. Lidar `points` $(N, 3)$, the grid `origin`
# $(3,)$ and `voxel_size` $(3,)$ are in the world frame, in metres. A point's
# voxel grid index is the integer triple
# $$(i, j, k) = \left\lfloor \frac{p - \text{origin}}{\text{voxel size}} \right\rfloor$$
# and the grid covers $0 \le i < \text{shape}_0$, and likewise for $j$ and $k$.

# %%
import numpy as np

# %%
# Baseline — Here the origin is $[0, 0, 0]$ and voxels are 1 m cubes, so the
# voxel grid index is just the floor of each coordinate.
points = np.array([[0.1, 0.1, 0.1], [0.9, 0.1, 0.1], [1.0, 0.0, 0.0]])
print("voxel coordinates:", np.floor(points).astype(int))
# Predict: These voxel grid indices, before running.
#
# Return sorted occupied voxel grid indices $(V, 3)$, `inverse` $(N,)$ ($-1$
# for points outside the grid), `counts` $(V,)$, and mean features $(V, C)$.
# Duplicate points must accumulate.

# %% [drill]
# Floor or truncate?
# With origin 0 and 0.5 m voxels, predict the index of $x = -0.25$ and
# $x = 0.25$ by `astype(int)` and by `np.floor`.
import numpy as np
scaled = (np.array([-0.25, 0.25]) - 0.) / 0.5
print(scaled.astype(int), np.floor(scaled).astype(int))

# %% [drill]
# The far edge of the grid
# Four 0.5 m voxels from origin 0 cover $[0, 2)$. Predict the indices of
# $x = 1.99$ and $x = 2.0$, and both inside tests.
import numpy as np
origin, voxel_size, shape = 0., 0.5, 4
upper = origin + shape * voxel_size
x = np.array([1.99, 2.0])
print(np.floor((x - origin) / voxel_size).astype(int), x < upper, x <= upper)

# %% [drill]
# Two points, one voxel
# Points 0 and 1 share voxel 0. Predict the per-voxel feature sums from
# fancy-index `+=` and from `np.add.at`.
import numpy as np
index = np.array([0, 0, 1])
features = np.array([1., 2., 3.])
lost = np.zeros(2)
lost[index] += features
kept = np.zeros(2)
np.add.at(kept, index, features)
print(lost, kept)

# %% [answer]
def voxel_pool(points: np.ndarray, features: np.ndarray, origin: np.ndarray,
               voxel_size: np.ndarray, shape: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pass

# %%
# Baseline check — Work a two-voxel example by hand, then run this cell.
from torch_judge import check
check("0066_voxel_pool_eval.py", stage="baseline")

# %%
# Follow-up — The grid is half-open: a point is inside when
# $\text{origin} \le p < \text{origin} + \text{shape} \cdot \text{voxel size}$
# on every axis. Points below the origin or on the MAX boundary are excluded.
# Empty input returns correctly shaped empty outputs. Permuting points and
# features together may reorder `inverse`, but not the pooled voxel map.
# Watch out: Below the origin, `(p - origin) / voxel_size` is negative, and
# `astype(int)` truncates toward zero: $-0.5$ becomes index 0, not $-1$. Use
# `np.floor`, then bound the integer voxel index itself: keep only
# $0 \le \text{index} < \text{shape}$. A metre test alone is not enough, since
# floor rounding can turn a point just below the upper edge into index `shape`.
#
# What policy should invalid metadata use? This exercise raises `ValueError`.
check("0066_voxel_pool_eval.py", stage="followup")

# %%
# Production challenge — Validate all shapes, real numeric dtypes, finite
# values, positive sizes and positive INTEGER grid dimensions. Never mutate
# inputs. Memory should scale with the input and the occupied voxels, not the
# grid volume. Use float64 for means; test a mostly empty, large grid.
# Why: Repeated-index assignment loses points because fancy-index `+=` writes
# each repeated voxel once, so only the last contribution survives;
# `np.add.at` applies every addition.
check("0066_voxel_pool_eval.py", stage="production")
