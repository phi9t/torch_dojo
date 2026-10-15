# %%
# 0131 · Historical points into the current ego frame in NumPy (practice set 7/7, 10 min)
# Goal: implement `transform_history`, vectorized over batch, agents and time,
# without mutating inputs. Column-vector convention: p_world = T @ p_local.

# %%
import numpy as np

# %%
# Baseline — points[B,A,T,3] are in their own historical ego frames;
# world_from_history[B,T,4,4]; world_from_current[B,4,4]. Return [B,A,T,3]
# in the current ego frame:
#   p_current = inv(world_from_current) @ world_from_history @ p_history
# Hand-checkable: history +10 on x, current +7, point [2,0,0] -> [5,0,0].
# History rotated +90 degrees about z, current identity, point [1,0,0] -> [0,1,0].

# %% [drill]
# Predict the translation example
# World x = 10 + 2 = 12; the current frame sits at 7. Predict the result.
print(12 - 7)

# %% [drill]
# Rigid inverse
# For T = [R | t], predict the inverse's rotation and translation, then confirm numerically.
R = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]); t = np.array([1., 2., 3.])
T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
inv = np.eye(4); inv[:3, :3] = R.T; inv[:3, 3] = -R.T @ t
print(np.allclose(inv @ T, np.eye(4)))

# %%
# Inputs conforming to the answer's signature.
def _pose(theta, t):
    c, s = np.cos(theta), np.sin(theta)
    P = np.eye(4); P[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]; P[:3, 3] = t
    return P

rng = np.random.default_rng(0)
B, A, T_ = 2, 3, 4
points: np.ndarray = rng.normal(size=(B, A, T_, 3))                       # [B,A,T,3]
world_from_history: np.ndarray = np.stack(
    [[_pose(0.1 * (b + t), (t, b, 0)) for t in range(T_)] for b in range(B)])  # [B,T,4,4]
world_from_current: np.ndarray = np.stack([_pose(0.3, (5, 5, 0)) for _ in range(B)])  # [B,4,4]

# %% [answer]
def transform_history(points: np.ndarray, world_from_history: np.ndarray,
                      world_from_current: np.ndarray) -> np.ndarray:
    # points [B,A,T,3]; world_from_history [B,T,4,4]; world_from_current [B,4,4]
    # returns [B,A,T,3] float64 in the current ego frame
    pass

# %%
from torch_judge import check
check("0131_history_to_ego_frame_eval.py", stage="baseline")

# %%
# Follow-up — Matches an explicit np.linalg.inv computation on random rigid
# poses, and transforming out then back recovers the points.
check("0131_history_to_ego_frame_eval.py", stage="followup")

# %%
# Production — No input mutation, batch independence, a large batch runs fast,
# mismatched shapes raise ValueError.
# Extension (not graded): invert a rigid transform with R^T and -R^T t instead
# of a general matrix inverse, and fold it into one einsum.
check("0131_history_to_ego_frame_eval.py", stage="production")
