# %%
# 0063 · Uniform oriented-box sampling, from local axes to volume checks
# Goal: Implement `sample_oriented_box(center, size, yaw, count, rng)` in
# NumPy: draw `count` points uniformly inside a yaw-rotated 3D box and return
# them in the world frame. Keep one answer and strengthen it at each stage.
#
# Practice scene: the parking lot. A parked car is a box with a centre in the
# world frame (metres, $z$ up) and a size along its own box-local axes: $+x$
# along the car's length (its heading), $+y$ to its left, $+z$ up, origin at
# the box centre. Yaw is in radians, counter-clockwise about $+z$ from world $+x$.
#
# This is an authored extension of a reported box-sampling question; the
# reported prompt did not establish 3D, yaw, or a particular RNG contract.

# %%
import numpy as np

# %%
# Baseline — For a box centred at world $[2, 0, 0]$ with size $[4, 2, 2]$ and
# yaw 0, box-local $x = -2$ maps to world $x = 0$.
center_probe = np.array([2., 0., 0.])
size_probe = np.array([4., 2., 2.])
print("local left face in world:", np.array([-2., 0., 0.]) + center_probe)
# Predict: Which interval may world $y$ occupy for this box?
#
# Implement `sample_oriented_box(center, size, yaw, count, rng)` for `center`
# $(3,)$, positive `size` $(3,)$, `yaw` in radians, a nonnegative integer
# `count` and an injected `np.random.Generator`, returning float
# $(\text{count}, 3)$ world points. Draw independent uniform box-local
# coordinates in $[-\tfrac12, \tfrac12] \cdot \text{size}$, then map them to the world.

# %% [drill]
# Row-stored points: which transpose?
# A box-local $+x$ point is stored as a row. Predict both results for a yaw of
# $+\pi/2$: which one points along world $+y$?
import numpy as np
yaw = np.pi / 2
c, s = np.cos(yaw), np.sin(yaw)
R = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
local = np.array([[1., 0., 0.]])
print(np.round(local @ R.T, 3) + 0.0, np.round(local @ R, 3) + 0.0)

# %% [drill]
# Degrees or radians?
# Predict $(\cos, \sin)$ of a yaw written as `90`, then as `np.pi / 2`.
import numpy as np
for yaw in (90, np.pi / 2):
    print(np.round([np.cos(yaw), np.sin(yaw)], 3) + 0.0)

# %% [drill]
# Rotate, then translate
# Map the box-local origin to the world for a box centred at $[2, 0, 0]$ with
# yaw $+\pi/2$. Predict both orders.
import numpy as np
c, s = np.cos(np.pi / 2), np.sin(np.pi / 2)
R = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
center = np.array([2., 0., 0.])
origin = np.zeros((1, 3))
print(np.round(origin @ R.T + center, 3) + 0.0, np.round((origin + center) @ R.T, 3) + 0.0)

# %% [answer]
# Revise this same implementation as the contracts grow.
def sample_oriented_box(center: np.ndarray, size: np.ndarray, yaw: float,
                        count: int, rng: np.random.Generator) -> np.ndarray:
    pass

# %%
from torch_judge import check
check("0063_oriented_box_sampling_eval.py", stage="baseline")

# %%
# Follow-up — Rotate about $z$ from box-local to world. For column vectors
# that is $p_{world} = R\,p_{local} + c$. Samples are stored as rows, so use
# `local @ R.T + center`. Invert with `(world - center) @ R`.
# Watch out: A box-local $+x$ point must land on world $+y$ after a $+\pi/2$
# yaw. Writing `local @ R` rotates by $-\text{yaw}$ instead, and still passes
# every check at yaw 0.
#
# Check seeded replay, inverse-transform bounds, and the population variance
# $\text{size}^2 / 12$ per axis, with a statistical tolerance, never exact equality.
quarter_turn = np.array([[np.cos(np.pi/2), -np.sin(np.pi/2), 0.],
                         [np.sin(np.pi/2), np.cos(np.pi/2), 0.],
                         [0., 0., 1.]])
print("quarter turn of local +x:", np.array([[1., 0., 0.]]) @ quarter_turn.T)
check("0063_oriented_box_sampling_eval.py", stage="followup")

# %%
# Production — `count=0` returns $(0, 3)$. Invalid shape, nonfinite geometry,
# nonpositive size, or a noninteger or negative count raises `ValueError`. Use
# only the supplied Generator; leave inputs unchanged. Generate
# $\text{count} \times 3$ values at once, so output and working memory grow
# $O(\text{count})$, without a Python loop per point.
# Predict: Why are random weights on the eight vertices not a uniform volume
# sampler?
check("0063_oriented_box_sampling_eval.py", stage="production")
