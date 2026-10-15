"""Staged reference and verifiers for Lloyd K-means."""

# %%
import numpy as np
from hypothesis import given, settings, strategies as st

# %%
# Drill 1: Fancy-index `+=` is buffered. NumPy first computes
# `sums[[0,0,0]] + v`, which is `[2, 3, 5]`, then assigns those three values to
# index 0 one after another, so the last write wins and it prints `[5.]`.
# `np.add.at` is unbuffered: every repeated destination accumulates, giving
# `[10.]`. Scattering points into cluster sums is exactly this case.
#
# Drill 2: `np.bincount` returns `max(labels) + 1` entries, so it prints
# `[2 1]` and the empty cluster silently disappears; the counts no longer line
# up with the $K$ centers. `minlength=3` gives `[2 1 0]`. Dividing sums by
# those counts prints `[0.5 9.  nan]`: $0/0$ is NaN, which is why an empty
# cluster must keep its previous center instead of taking a computed mean.
#
# Drill 3: The first assignment is `[0 1 1 1]`, and the update moves center 1
# to the mean of 1, 9 and 10, `6.667`. Point 1 is now 1 away from center 0
# but 5.667 from center 1, so reassigning gives `[0 0 1 1]`. Returning the
# first labels with the new centers pairs point 1 with a center that is no
# longer nearest, and inertia from them would count $5.667^2 \approx 32.1$
# instead of 1 for that point.

# %%
def _assign(X: np.ndarray, centers: np.ndarray, block_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Keep only one [block_size,D] difference buffer, even when K is large."""
    labels = np.empty(len(X), dtype=np.int64)
    nearest = np.empty(len(X), dtype=np.float64)
    for start in range(0, len(X), block_size):
        points = X[start:start + block_size]
        best = np.full(len(points), np.inf)
        winners = np.zeros(len(points), dtype=np.int64)
        for index, center in enumerate(centers):
            delta = points - center
            distance = np.einsum("nd,nd->n", delta, delta)
            improved = distance < best  # Strict comparison retains the lowest-index tie.
            best[improved] = distance[improved]
            winners[improved] = index
        labels[start:start + len(points)] = winners
        nearest[start:start + len(points)] = best
    if not np.isfinite(nearest).all():
        raise ValueError("squared-distance overflow")
    return labels, nearest


def kmeans(X: np.ndarray, initial_centers: np.ndarray, max_iter: int = 100,
           *, block_size: int = 1024) -> tuple[np.ndarray, np.ndarray, float]:
    raw_X, raw_centers = np.asarray(X), np.asarray(initial_centers)
    if (raw_X.ndim != 2 or raw_centers.ndim != 2 or not len(raw_X)
            or not len(raw_centers) or raw_X.shape[1] == 0
            or raw_X.shape[1] != raw_centers.shape[1]
            or raw_X.dtype.kind not in "iuf" or raw_centers.dtype.kind not in "iuf"
            or isinstance(max_iter, (bool, np.bool_))
            or not isinstance(max_iter, (int, np.integer)) or max_iter < 0
            or isinstance(block_size, (bool, np.bool_))
            or not isinstance(block_size, (int, np.integer)) or block_size < 1):
        raise ValueError("need nonempty numeric [N,D] points, [K,D] centers, and valid iteration/block limits")
    points = raw_X.astype(np.float64)
    centers = raw_centers.astype(np.float64, copy=True)
    if not np.isfinite(points).all() or not np.isfinite(centers).all():
        raise ValueError("finite inputs required")
    for _ in range(max_iter):
        labels, _ = _assign(points, centers, block_size)
        updated = centers.copy()
        # Repeated labels are repeated destinations: indexed += buffers its
        # reads and does not accumulate every point. ufunc.at does.
        sums = np.zeros_like(centers)
        counts = np.zeros(len(centers), dtype=np.int64)
        np.add.at(sums, labels, points)
        np.add.at(counts, labels, 1)
        occupied = counts > 0
        updated[occupied] = sums[occupied] / counts[occupied, None]
        if np.array_equal(updated, centers):
            break
        centers = updated
    labels, distance = _assign(points, centers, block_size)
    inertia = float(distance.sum())
    if not np.isfinite(inertia):
        raise ValueError("inertia overflow")
    return centers, labels, inertia


# %% [verifiers]
def verify_baseline_simple_clusters(kmeans):
    X = np.array([[0.0], [1.0], [9.0], [10.0]])
    c, l, i = kmeans(X, np.array([[0.0], [10.0]]))
    assert np.allclose(c.ravel(), [0.5, 9.5]) and np.array_equal(l, [0, 0, 1, 1]) and np.isclose(i, 1.0)


def verify_followup_empty_center_stays_and_labels_correspond(kmeans):
    X = np.array([[0.0], [0.0], [2.0]])
    c, l, i = kmeans(X, np.array([[0.0], [2.0], [50.0]]))
    assert np.isclose(c[2, 0], 50.0) and np.array_equal(l, ((X[:, None] - c[None]) ** 2).sum(2).argmin(1))


def verify_baseline_k_one_and_no_mutation(kmeans):
    X = np.array([[1.0], [3.0]])
    start = np.array([[0.0]])
    c, l, i = kmeans(X, start)
    assert np.isclose(c[0, 0], 2.0) and np.array_equal(start, [[0.0]]) and np.isclose(i, 2.0)


def verify_followup_repeated_destinations_accumulate_all_points(kmeans):
    points = np.array([[2., 1.], [3., 4.], [5., 7.]])
    start = np.array([[0., 0.], [100., 100.]])
    centers, labels, inertia = kmeans(points, start, max_iter=1)
    assert np.allclose(centers, [[10/3, 4.], [100., 100.]])
    assert np.array_equal(labels, [0, 0, 0])
    assert np.isclose(inertia, 68/3)


def verify_followup_zero_iterations_and_ties(kmeans):
    X = np.array([[0.0], [2.0]])
    start = np.array([[1.0], [1.0]])
    centers, labels, inertia = kmeans(X, start, max_iter=0)
    assert np.array_equal(centers, start)
    assert np.array_equal(labels, [0, 0])
    assert inertia == 2.0


def verify_production_small_assignment_batches(kmeans):
    X = np.array([[0., 0.], [1., 0.], [9., 0.], [10., 0.]])
    start = np.array([[0., 0.], [10., 0.]])
    expected = kmeans(X, start)
    actual = kmeans(X, start, block_size=1)
    assert np.allclose(actual[0], expected[0])
    assert np.array_equal(actual[1], expected[1])
    assert np.isclose(actual[2], expected[2])


def verify_production_validation_and_large_offset(kmeans):
    for points, centers, kwargs in (
        (np.empty((0, 1)), np.zeros((1, 1)), {}),
        (np.zeros((2, 1)), np.zeros((0, 1)), {}),
        (np.zeros((2, 1)), np.zeros((1, 2)), {}),
        (np.array([[np.inf]]), np.zeros((1, 1)), {}),
        (np.zeros((2, 1)), np.zeros((1, 1)), {"max_iter": -1}),
        (np.zeros((2, 1)), np.zeros((1, 1)), {"block_size": 0}),
    ):
        try:
            kmeans(points, centers, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid input accepted")
    X = np.array([[1e9], [1e9 + 1], [1e9 + 9], [1e9 + 10]])
    centers, labels, inertia = kmeans(X, np.array([[1e9], [1e9 + 10]]), block_size=2)
    assert np.allclose(centers.ravel() - 1e9, [.5, 9.5])
    assert np.array_equal(labels, [0, 0, 1, 1]) and np.isclose(inertia, 1)


def verify_production_generated_batch_equivalence(kmeans):
    @settings(max_examples=20, derandomize=True, deadline=None)
    @given(st.lists(st.integers(-30, 30), min_size=2, max_size=20))
    def property_check(values):
        points = np.asarray(values, dtype=float)[:, None]
        initial = np.array([[-20.], [20.]])
        original = points.copy()
        a = kmeans(points, initial, max_iter=4, block_size=1)
        b = kmeans(points, initial, max_iter=4, block_size=7)
        assert np.allclose(a[0], b[0])
        assert np.array_equal(a[1], b[1])
        assert np.isclose(a[2], b[2])
        assert np.array_equal(points, original)
    property_check()


# %%
from torch_judge import evaluate
if not evaluate("0061_kmeans_eval.py", kmeans):
    raise SystemExit(1)
