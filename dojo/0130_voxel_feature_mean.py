# %%
# 0130 · Point-to-voxel feature averaging in NumPy (practice set 6/7, 15 min)
# Goal: implement `voxel_mean` with no Python loop over points (NumPy unique,
# add.at or bincount are allowed).

# %%
import numpy as np

# %%
# Baseline — points[N,3] finite, features[N,C] finite, origin[3], voxel_size > 0,
# grid_shape three positive ints.
# Voxel index = floor((point - origin) / voxel_size). Keep points with
# 0 <= index < grid_shape on EVERY axis (upper boundary excluded).
# Return (coords[K,3] int64, means[K,C] float64, counts[K] int64), coordinates
# sorted lexicographically by x, then y, then z. Empty -> correctly shaped empties.
# Hand-checkable: x=[0.1,0.9,1.0,2.0], features [[2],[4],[8],[99]], voxel_size 1,
# grid (2,1,1) -> coords [[0,0,0],[1,0,0]], means [[3],[8]], counts [2,1].

# %% [drill]
# floor or int()?
# Predict both for -0.1 / 1.0. Which one puts the point out of bounds?
print(np.floor(-0.1), int(-0.1))

# %% [drill]
# Sorted order for free
# ravel_multi_index of (x, y, z) in a (2, 2, 2) grid: predict the flat ids of
# (0,1,1) and (1,0,0), and which comes first.
print(np.ravel_multi_index((0, 1, 1), (2, 2, 2)), np.ravel_multi_index((1, 0, 0), (2, 2, 2)))

# %% [drill]
# Cast before or after the bounds check?
# Predict what astype(int64) does to 1e300. (Compare on the float index first.)
with np.errstate(invalid="ignore"):
    print(np.array([1e300]).astype(np.int64))

# %%
# Inputs conforming to the answer's signature.
rng = np.random.default_rng(0)
points: np.ndarray = rng.uniform(-1, 5, size=(12, 3))   # [N,3]
features: np.ndarray = rng.normal(size=(12, 2))         # [N,C]
origin: np.ndarray = np.array([0.5, -0.2, 0.1])         # [3]
voxel_size: float = 0.7
grid_shape: tuple[int, int, int] = (5, 4, 6)

# %% [answer]
def voxel_mean(points: np.ndarray, features: np.ndarray, origin: np.ndarray,
               voxel_size: float, grid_shape: tuple[int, int, int]
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # points [N,3]; features [N,C]; origin [3]
    # returns (coords [K,3] int64, means [K,C] float64, counts [K] int64)
    pass

# %%
from torch_judge import check
check("0130_voxel_feature_mean_eval.py", stage="baseline")

# %%
# Follow-up — Nonzero origin, negative coordinates, voxel boundaries, several
# channels, and a dictionary-based reference on random points.
check("0130_voxel_feature_mean_eval.py", stage="followup")

# %%
# Production — Huge coordinates must not overflow an int cast, bad inputs raise
# ValueError, inputs are not mutated, and 200k points finish quickly.
# Extension (not graded): also return each voxel's mean point coordinate; then
# describe how you would process a stream in chunks (carry per-voxel sum and
# count, divide once at the end).
check("0130_voxel_feature_mean_eval.py", stage="production")
