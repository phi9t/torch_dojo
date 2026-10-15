"""Reference and cases for Point in Rotated Box."""

# %%
import numpy as np

# %%
# Drill 1: The first line prints `[[1.5 0.  0. ]] True`, the second
# `[[-1.5 -3.   0. ]] False`. The rotation turns about the BOX centre, so the
# world displacement $p - c$ must be formed before the inverse rotation:
# $R^T (p - c)$. Rotating the raw world point first spins it about the world
# origin, and subtracting the centre afterwards no longer cancels it, so a
# point on the car's long axis looks 3 m away to the side.
#
# Drill 2: With the inverse rotation it prints `[ True False]`; without it,
# `[False  True]`. Yaw $+\pi/2$ turns box-local $+x$, the 4 m side, onto world
# $+y$. So $[0, 1.5, 0]$ lies along the car's length (box-local $x = 1.5$,
# inside the half-length 2), and $[1.5, 0, 0]$ sticks out of its side
# (box-local $y = -1.5$, beyond the half-width 1). Skipping the rotation
# compares world axes with box-local sizes and swaps the answers.
#
# Drill 3: It prints `2.0000000000000004 False True`. The point is exactly on
# the $+x$ face in box-local coordinates, but $\cos(\pi/6)$ and $\sin(\pi/6)$
# are rounded, and the world-to-box-local round trip leaves $x$ one ulp past 2.
# A bare `<=` drops a point that is on the face; the absolute $10^{-10}$
# tolerance keeps it while still rejecting a point $2 \times 10^{-9}$ outside.

# %%
def point_in_rotated_box(points: np.ndarray, centers: np.ndarray, sizes: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    points, centers, sizes, yaw = map(np.asarray, (points, centers, sizes, yaw))
    if (points.ndim != 2 or points.shape[1] != 3 or centers.ndim != 2
            or centers.shape[1] != 3 or sizes.shape != centers.shape
            or yaw.shape != (len(centers),)
            or any(arr.dtype.kind not in 'fiu' for arr in (points, centers, sizes, yaw))
            or not np.isfinite(points).all() or not np.isfinite(centers).all()
            or not np.isfinite(sizes).all() or not np.isfinite(yaw).all()
            or np.any(sizes <= 0)):
        raise ValueError("invalid points or boxes")
    result = np.empty((len(points), len(centers)), dtype=bool)
    cosine, sine = np.cos(yaw), np.sin(yaw)
    block = 256
    # The NxM result is required, while temporary pair arrays stay bounded.
    for i in range(0, len(points), block):
        pp = points[i:i + block]
        for j in range(0, len(centers), block):
            cc, ss = centers[j:j + block], sizes[j:j + block]
            dx = pp[:, None, 0] - cc[None, :, 0]
            dy = pp[:, None, 1] - cc[None, :, 1]
            dz = pp[:, None, 2] - cc[None, :, 2]
            c, s = cosine[None, j:j + block], sine[None, j:j + block]
            x = dx * c + dy * s
            y = -dx * s + dy * c
            result[i:i + block, j:j + block] = (
                (np.abs(x) <= ss[None, :, 0] / 2 + 1e-10)
                & (np.abs(y) <= ss[None, :, 1] / 2 + 1e-10)
                & (np.abs(dz) <= ss[None, :, 2] / 2 + 1e-10))
    return result


# %% [verifiers]
def verify_baseline_axis_aligned_membership(point_in_rotated_box):
    points = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
    result = point_in_rotated_box(points, np.zeros((1, 3)),
                                  np.array([[2., 1., 2.]]), np.zeros(1))
    assert result.shape == (3, 1) and result.dtype == np.bool_
    assert np.array_equal(result[:, 0], [True, True, False])


def verify_followup_rotated_membership(point_in_rotated_box):
    points = np.array([[1., 0., 0.], [0., 1., 0.], [2., 0., 0.]])
    centers = np.zeros((2, 3))
    sizes = np.array([[2., 1., 2.], [2., 1., 2.]])
    result = point_in_rotated_box(points, centers, sizes,
                                  np.array([0., np.pi / 2]))
    assert result.shape == (3, 2)
    assert np.array_equal(result, [[True, False], [False, True], [False, False]])


def verify_followup_empty_dimensions(point_in_rotated_box):
    assert point_in_rotated_box(np.zeros((0, 3)), np.zeros((2, 3)), np.ones((2, 3)), np.zeros(2)).shape == (0, 2)
    assert point_in_rotated_box(np.zeros((1, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)).shape == (1, 0)


def verify_followup_translation_and_invalid_size(point_in_rotated_box):
    assert point_in_rotated_box(np.array([[5.0, 6.0, 7.0]]), np.array([[5.0, 6.0, 7.0]]), np.ones((1, 3)), np.zeros(1))[0, 0]
    try:
        point_in_rotated_box(np.zeros((1, 3)), np.zeros((1, 3)), np.zeros((1, 3)), np.zeros(1))
    except ValueError:
        pass
    else:
        raise AssertionError('invalid size accepted')


def verify_production_faces_and_inverse_transform(point_in_rotated_box):
    center = np.array([[3., -2., 1.]])
    size = np.array([[4., 2., 6.]])
    yaw = np.array([np.pi / 3])
    c, s = np.cos(yaw[0]), np.sin(yaw[0])
    rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    local = np.array([[2., 0., 0.], [0., -1., 0.], [0., 0., 3.],
                      [2. + 5e-11, 0., 0.], [2. + 2e-9, 0., 0.]])
    world = local @ rotation.T + center[0]
    result = point_in_rotated_box(world, center, size, yaw)
    assert np.array_equal(result[:, 0], [True, True, True, True, False])


def verify_production_invalid_shapes_and_immutability(point_in_rotated_box):
    points = np.array([[1., 2., 3.]])
    centers = np.array([[0., 0., 0.]])
    sizes = np.ones((1, 3))
    yaw = np.zeros(1)
    originals = [item.copy() for item in (points, centers, sizes, yaw)]
    point_in_rotated_box(points, centers, sizes, yaw)
    assert all(np.array_equal(item, original) for item, original in zip((points, centers, sizes, yaw), originals))
    for args in (
        (np.zeros((1, 2)), centers, sizes, yaw),
        (points, centers, sizes, np.zeros(2)),
        (points, centers, np.zeros((1, 2)), yaw),
        (points, centers, sizes, np.array([np.nan])),
    ):
        try:
            point_in_rotated_box(*args)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid geometry accepted")


def verify_production_large_grid_and_translated_membership(point_in_rotated_box):
    points = np.zeros((513, 3))
    centers = np.zeros((257, 3))
    sizes = np.ones((257, 3))
    yaw = np.linspace(-np.pi, np.pi, 257)
    inside = point_in_rotated_box(points, centers, sizes, yaw)
    shifted = point_in_rotated_box(points + [5., -7., 2.], centers + [5., -7., 2.], sizes, yaw)
    assert inside.shape == (513, 257) and inside.dtype == np.bool_
    assert inside.all() and np.array_equal(inside, shifted)


# %%
from torch_judge import evaluate
if not evaluate("0064_point_in_rotated_box_eval.py", point_in_rotated_box):
    raise SystemExit(1)
