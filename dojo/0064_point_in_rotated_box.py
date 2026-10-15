# %%
# 0064 · Point in rotated box, from one box to a pair grid
# Goal: Implement `point_in_rotated_box(points, centers, sizes, yaw)` in NumPy:
# for $N$ world points and $M$ yaw-rotated 3D boxes return the bool $(N, M)$
# membership grid. Keep one answer and strengthen it at each stage.
#
# Practice scene: the parking lot. Lidar points `points` $(N, 3)$ and box
# `centers` $(M, 3)$ are in the world frame (metres, $z$ up); `sizes` $(M, 3)$
# are strictly positive extents along each box-local frame (origin at the box
# centre, $+x$ along the car's length, $+y$ to its left, $+z$ up); `yaw` $(M,)$ is in radians,
# counter-clockwise about $+z$. A point on a face counts as inside, with an
# absolute box-local tolerance of $10^{-10}$.

# %%
import numpy as np

# %%
# Baseline — At yaw 0, a box of size $[2, 2, 2]$ centred at the world origin
# spans $[-1, 1]$ on every axis.
probe_points = np.array([[0., 0., 0.], [1., 0., 0.], [1.1, 0., 0.]])
print("axis-aligned membership:", np.all(np.abs(probe_points) <= 1., axis=1))
# Predict: The three memberships, before running the probe.
#
# Write `point_in_rotated_box(points, centers, sizes, yaw)`.

# %% [drill]
# Subtract the centre first
# A box at world $[3, 0, 0]$ with yaw $+\pi/2$ and size $[4, 2, 2]$; the world
# point $[3, 1.5, 0]$. Predict both box-local coordinates and memberships.
import numpy as np
c, s = np.cos(np.pi / 2), np.sin(np.pi / 2)
R = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
center = np.array([3., 0., 0.])
half = np.array([2., 1., 1.])
point = np.array([[3., 1.5, 0.]])
right = (point - center) @ R
wrong = point @ R - center
print(np.round(right, 3) + 0.0, np.all(np.abs(right) <= half))
print(np.round(wrong, 3) + 0.0, np.all(np.abs(wrong) <= half))

# %% [drill]
# Where does the long side point?
# A car of size $[4, 2, 2]$ at the world origin is turned by yaw $+\pi/2$.
# Predict which world point is inside, with and without the inverse rotation.
import numpy as np
c, s = np.cos(np.pi / 2), np.sin(np.pi / 2)
R = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
half = np.array([2., 1., 1.])
world = np.array([[0., 1.5, 0.], [1.5, 0., 0.]])
print(np.all(np.abs(world @ R) <= half, axis=1))
print(np.all(np.abs(world) <= half, axis=1))

# %% [drill]
# Exactly on the face
# Place a point on the box-local $+x$ face ($x = 2$), map it to the world at
# yaw $\pi/6$, and map it back. Predict the recovered $x$ and both tests.
import numpy as np
c, s = np.cos(np.pi / 6), np.sin(np.pi / 6)
R = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
center = np.array([3., -2., 1.])
world = np.array([[2., 0., 0.]]) @ R.T + center
x = float(((world - center) @ R)[0, 0])
print(x, abs(x) <= 2., abs(x) <= 2. + 1e-10)

# %% [answer]
# Keep one answer cell; add helpers here if useful.
def point_in_rotated_box(points: np.ndarray, centers: np.ndarray,
                         sizes: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    pass

# %%
from torch_judge import check
check("0064_point_in_rotated_box_eval.py", stage="baseline")

# %%
# Follow-up — Go from world to box-local: subtract the centre, then apply the
# inverse rotation. If $p_{world} = R\,p_{local} + c$, then
# $p_{local} = R^T (p_{world} - c)$, which for row-stored points is
# `(world - center) @ R`. Include faces, translate correctly, and return the
# right empty shapes $(0, M)$ and $(N, 0)$.
# Watch out: At yaw $+\pi/2$ the long box-local $x$ axis points along world
# $+y$. Comparing the world displacement against `sizes` without rotating
# tests a different box.
check("0064_point_in_rotated_box_eval.py", stage="followup")

# %%
# Production — Reject invalid shapes, nonfinite coordinates, or nonpositive
# sizes with `ValueError`; never mutate inputs. A full bool $(N, M)$ output is
# required, but a full $(N, M, 3)$ displacement tensor is avoidable. Process
# point/box blocks, computing `dx`, `dy`, `dz` as 2D arrays. This uses
# $O(NM)$ output plus $O(\text{block}^2)$ temporary memory. Test simultaneous
# translation, rotated faces just inside and outside the tolerance, and a grid
# crossing blocks.
# Why: Faces need the $10^{-10}$ tolerance because the world-to-box-local
# round trip rounds; a point built on a face can come back a few ulps outside.
check("0064_point_in_rotated_box_eval.py", stage="production")
