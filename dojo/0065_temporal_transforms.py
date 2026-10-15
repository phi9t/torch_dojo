# %%
# 0065 · Temporal ego coordinates
# Goal: Implement `to_current_ego(trajectories, world_from_t, world_from_current, valid)`
# in NumPy: move every actor's past positions, each recorded in the ego frame
# at its own time $t$, into the current ego frame.
#
# Practice scene: the parking lot, observed over $T$ sweeps while the ego car
# drives through it. Three frames meet here: the ego frame at time $t$ (where
# each sweep was recorded), the world frame (fixed, metres, $z$ up), and the
# current ego frame (the latest sweep). Each ego frame has its origin on the
# car, $+x$ forward, $+y$ left and $+z$ up; a pose's yaw is in radians,
# counter-clockwise about $+z$. `trajectories` $(B, A, T, 3)$ holds
# points in the ego frame at time $t$. The rigid poses `world_from_t`
# $(B, T, 4, 4)$ and `world_from_current` $(B, 4, 4)$ map ego column vectors
# into the world. `valid` $(B, A, T)$ marks real points; invalid outputs are 0.
# A pose is a rotation $R$ and a translation $d$ (the ego origin in world
# coordinates); $t$ always means time.
#
# Authored practice variant; see `practice_bank/drills.tsv`.

# %%
import numpy as np

# %%
# Worked probe: the point $(1, 0, 0)$ in the ego frame at time $t$. That ego
# was yawed $\pi/2$ radians and sat at world $x = 5$. The current ego sits at
# world $x = 2$ with no yaw.
# Predict: The point in the world, then in the current ego frame.
#
# Rotating $(1, 0, 0)$ by $\pi/2$ gives $(0, 1, 0)$, plus the old ego's offset
# gives world $(5, 1, 0)$; removing the current ego's offset gives $(3, 1, 0)$.
print('frame chain: inverse(T_world_from_current) @ T_world_from_t')
# Why: Read a pose name right to left. `world_from_t` takes a point in the ego
# frame at time $t$ to the world, and its inverse `current_from_world` takes a
# world point to the current ego frame, so
# $$T_{\text{current}\leftarrow t} = T_{\text{world}\leftarrow\text{current}}^{-1}\, T_{\text{world}\leftarrow t}$$
# The inner frame names must match where the two transforms meet.

# %% [drill]
# Which order composes the frames?
# Use the worked probe's poses. Predict the point after
# `current_from_world @ world_from_t`, then after the reversed product.
import numpy as np
world_from_t = np.eye(4)
world_from_t[:2, :2] = [[0., -1.], [1., 0.]]
world_from_t[0, 3] = 5.
world_from_current = np.eye(4)
world_from_current[0, 3] = 2.
current_from_world = np.linalg.inv(world_from_current)
p = np.array([1., 0., 0., 1.])
print((current_from_world @ world_from_t @ p)[:3])
print((world_from_t @ current_from_world @ p)[:3])

# %% [drill]
# Inverting a rigid pose
# The current ego is yawed $+\pi/2$ and sits at world $[2, 0, 0]$. Predict the
# world point $[2, 1, 0]$ in the current ego frame, computed as
# $R^T (w - d)$ and as $R^T w - d$, where $d$ is the ego's world position.
import numpy as np
R = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
d = np.array([2., 0., 0.])
w = np.array([2., 1., 0.])
print(R.T @ (w - d), R.T @ w - d)

# %% [drill]
# Masking padded NaNs
# Row 2 is a padded slot full of NaN. Predict both masked results.
import numpy as np
points = np.array([[1., 2., 3.], [np.nan, np.nan, np.nan]])
valid = np.array([True, False])
print(points * valid[:, None])
print(np.where(valid[:, None], points, 0.))

# %%
# Baseline — Accept float32 or float64 finite trajectories and proper rigid
# poses; return $(B, A, T, 3)$ points in the current ego frame. With identity
# poses, keep valid points and zero the padded ones.

# %% [answer]
def to_current_ego(trajectories: np.ndarray, world_from_t: np.ndarray,
                   world_from_current: np.ndarray, valid: np.ndarray) -> np.ndarray:
    pass

# %%
from torch_judge import check
check('0065_temporal_transforms_eval.py', stage='baseline')

# %%
# Follow-up — Compose time-varying poses in the named order: ego at time $t$
# to world, then world to current ego. Do not mix actors or batches.
# Watch out: The inverse of a rigid pose $(R, d)$ is $(R^T, -R^T d)$, not
# $(R^T, -d)$. Rotate the difference $w - d$, not the world point alone.
#
# Points already in the world frame would need only the second step.
check('0065_temporal_transforms_eval.py', stage='followup')

# %%
# Production — Reject shape, dtype, mask, nonfinite valid data and nonrigid
# poses with `ValueError`. Proper rigid means $R^T R = I$, $\det R = 1$ and a
# bottom row of $[0, 0, 0, 1]$. Preserve float32/float64 dtype, support empty
# dimensions and strided input, and do not mutate inputs. Vectorize across
# $B$, $A$ and $T$; $O(BAT)$ work and output.
# Watch out: Invalid coordinates may be NaN and must not leak. Multiplying by
# the mask keeps NaN, since $0 \cdot \text{NaN} = \text{NaN}$; select with `np.where`.
check('0065_temporal_transforms_eval.py', stage='production')
