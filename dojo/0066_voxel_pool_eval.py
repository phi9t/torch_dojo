"""Three-stage reference for sparse voxel indices and mean features."""

# %%
import numpy as np
from hypothesis import given, settings, strategies as st

# %%
# Drill 1: It prints `[0 0] [-1  0]`. The scaled coordinates are $-0.5$ and
# $0.5$. `astype(int)` truncates toward zero, so $-0.5$ becomes 0 and a point a
# quarter metre BELOW the origin is pooled with points inside the first voxel.
# The floor rounds toward $-\infty$ and gives $-1$, an index outside the grid,
# which the bounds test then rejects. Truncation and floor only agree for
# nonnegative values, which is why the bug survives tests that stay inside.
#
# Drill 2: It prints `[3 4] [ True False] [ True  True]`. Four voxels own
# indices 0 to 3, so the far edge $x = 2.0$ would be index 4, one past the end.
# Each voxel is half-open, $[i \cdot 0.5, (i + 1) \cdot 0.5)$, and the grid is
# their union $[0, 2)$; the strict `<` test excludes the edge point. With `<=`
# the point is accepted and then either indexes out of range or, if clamped,
# is silently added to voxel 3. The metre test alone is still not a safe
# guard: $(x - \text{origin}) / \text{voxel size}$ rounds, so a point a hair
# below the upper edge can pass `x < upper` and still floor to index `shape`
# (e.g. origin $-10.8$, 0.2 m voxels, shape 44). Bound the integer index too,
# keeping only $0 \le \text{index} < \text{shape}$.
#
# Drill 3: Fancy-index `+=` prints `[2. 3.]`; `np.add.at` prints `[3. 3.]`.
# `lost[index] += features` is `lost[index] = lost[index] + features`: the
# right side is computed from a gathered copy, then written back once per
# index, so for the repeated index 0 the last write (0 + 2) wins and the 1 is
# lost. `np.add.at` is unbuffered and applies every addition in turn, so
# voxel 0 receives $1 + 2 = 3$.

# %%
def voxel_pool(points: np.ndarray, features: np.ndarray, origin: np.ndarray,
               voxel_size: np.ndarray, shape: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points)
    features = np.asarray(features)
    origin = np.asarray(origin)
    voxel_size = np.asarray(voxel_size)
    shape = np.asarray(shape)
    if (points.ndim != 2 or points.shape[1] != 3 or features.ndim != 2
            or len(features) != len(points) or origin.shape != (3,)
            or voxel_size.shape != (3,) or shape.shape != (3,)):
        raise ValueError("need points[N,3], features[N,C], and three-vectors")
    if any(x.dtype.kind not in "iuf" for x in (points, features, origin, voxel_size)) or shape.dtype.kind not in "iu":
        raise ValueError("real numeric inputs and integer grid shape required")
    points = points.astype(np.float64)
    features = features.astype(np.float64)
    origin = origin.astype(np.float64)
    voxel_size = voxel_size.astype(np.float64)
    if (not all(np.isfinite(x).all() for x in (points, features, origin, voxel_size))
            or np.any(voxel_size <= 0) or np.any(shape <= 0)):
        raise ValueError("finite data, positive sizes, and positive shape required")
    upper = origin + shape.astype(np.float64) * voxel_size
    if not np.isfinite(upper).all():
        raise ValueError("grid extent overflows")
    inside = np.all((points >= origin) & (points < upper), axis=1)
    indices = np.floor((points[inside] - origin) / voxel_size).astype(np.int64)
    indices = np.minimum(indices, shape - 1)
    unique, local_inverse, counts = np.unique(indices, axis=0, return_inverse=True, return_counts=True)
    inverse = np.full(len(points), -1, dtype=np.int64)
    inverse[inside] = local_inverse
    sums = np.zeros((len(unique), features.shape[1]), dtype=np.float64)
    np.add.at(sums, local_inverse, features[inside])
    means = sums / counts[:, None]
    return unique, inverse, counts, means


# %% [verifiers]
def verify_baseline_repeated_points(voxel_pool):
    p = np.array([[.1, .1, .1], [.9, .1, .1], [1., 0., 0.]])
    f = np.array([[1., 2.], [3., 4.], [5., 6.]])
    c, inv, n, m = voxel_pool(p, f, np.zeros(3), np.ones(3), np.array([2, 2, 2]))
    assert np.array_equal(c, [[0, 0, 0], [1, 0, 0]])
    assert np.array_equal(inv, [0, 0, 1]) and np.array_equal(n, [2, 1])
    assert np.allclose(m, [[2, 3], [5, 6]])


def verify_baseline_sorted_coordinates(voxel_pool):
    p = np.array([[2.1, 0., 0.], [.1, 0., 0.], [1.1, 0., 0.]])
    c, inv, n, _ = voxel_pool(p, np.ones((3, 1)), np.zeros(3), np.ones(3), np.array([3, 1, 1]))
    assert np.array_equal(c[:, 0], [0, 1, 2])
    assert np.array_equal(inv, [2, 0, 1]) and np.array_equal(n, [1, 1, 1])


def verify_followup_half_open_and_negative(voxel_pool):
    p = np.array([[-.01, 0, 0], [2., 0, 0], [1.99, 0, 0]])
    c, inv, n, _ = voxel_pool(p, np.ones((3, 1)), np.zeros(3), np.ones(3), np.array([2, 2, 2]))
    assert np.array_equal(inv, [-1, -1, 0]) and np.array_equal(c, [[1, 0, 0]])
    assert n.sum() == 1


def verify_followup_empty_and_permutation(voxel_pool):
    empty = voxel_pool(np.zeros((0, 3)), np.zeros((0, 2)), np.zeros(3), np.ones(3), np.ones(3, dtype=int))
    assert [x.shape for x in empty] == [(0, 3), (0,), (0,), (0, 2)]
    p = np.array([[0., 0., 0.], [1., 0., 0.], [.2, 0., 0.]])
    f = np.array([[1.], [2.], [3.]])
    a = voxel_pool(p, f, np.zeros(3), np.ones(3), np.array([2, 1, 1]))
    b = voxel_pool(p[::-1], f[::-1], np.zeros(3), np.ones(3), np.array([2, 1, 1]))
    assert np.array_equal(a[0], b[0]) and np.allclose(a[3], b[3])


def verify_production_validation_and_immutability(voxel_pool):
    p = np.array([[.5, .5, .5]])
    f = np.array([[2.]])
    copies = [x.copy() for x in (p, f)]
    voxel_pool(p, f, np.zeros(3), np.ones(3), np.array([2, 2, 2]))
    assert np.array_equal(p, copies[0]) and np.array_equal(f, copies[1])
    bad = [(p, f, np.array([np.nan, 0., 0.]), np.ones(3), np.array([2, 2, 2])),
           (p, f, np.zeros(3), np.array([0., 1., 1.]), np.array([2, 2, 2])),
           (p, f, np.zeros(3), np.ones(3), np.array([2., 2., 2.])),
           (p, np.array([[np.inf]]), np.zeros(3), np.ones(3), np.array([2, 2, 2]))]
    for args in bad:
        try:
            voxel_pool(*args)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid voxel input accepted")


def verify_production_large_sparse_grid(voxel_pool):
    p = np.array([[0., 0., 0.], [199., 199., 199.], [0.5, 0., 0.]])
    c, inv, n, m = voxel_pool(p, np.array([[1.], [3.], [5.]]), np.zeros(3), np.ones(3), np.array([200, 200, 200]))
    assert c.shape == (2, 3) and np.array_equal(n, [2, 1])
    assert np.array_equal(inv, [0, 1, 0]) and np.allclose(m[:, 0], [3., 3.])


def verify_production_generated_count_conservation(voxel_pool):
    @settings(max_examples=30, deadline=None, derandomize=True)
    @given(st.lists(st.tuples(st.integers(-1, 4), st.integers(-1, 4)), max_size=20))
    def property_check(raw):
        p = np.array([(x, y, 0.) for x, y in raw], dtype=float).reshape(-1, 3)
        f = np.ones((len(p), 1))
        c, inv, n, m = voxel_pool(p, f, np.zeros(3), np.ones(3), np.array([4, 4, 1]))
        assert n.sum() == np.count_nonzero(inv >= 0)
        assert np.array_equal(n.astype(float), m[:, 0] * n)
        assert len(c) == len(n)

    property_check()


# %%
from torch_judge import evaluate
if not evaluate("0066_voxel_pool_eval.py", voxel_pool):
    raise SystemExit(1)
