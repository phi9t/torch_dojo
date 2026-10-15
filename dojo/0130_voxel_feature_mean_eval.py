"""Reference and staged checks for point-to-voxel feature averaging (NumPy)."""
# Drill answers: floor, not truncation, so -0.1 lands in voxel -1 (out of
# bounds) rather than voxel 0. Check bounds on the float index BEFORE casting
# to int, or 1e300 overflows. ravel_multi_index order == lexicographic x,y,z.

# %%
import numpy as np


# %%
def voxel_mean(points, features, origin, voxel_size, grid_shape):
    points = np.asarray(points, dtype=np.float64)
    features = np.asarray(features, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64)
    shape = tuple(grid_shape)
    if (points.ndim != 2 or points.shape[1] != 3 or features.ndim != 2 or len(features) != len(points)
            or origin.shape != (3,) or len(shape) != 3):
        raise ValueError("need points[N,3], features[N,C], origin[3], grid_shape of 3")
    if not np.isfinite(points).all() or not np.isfinite(features).all() or not np.isfinite(origin).all():
        raise ValueError("points, features and origin must be finite")
    if not voxel_size > 0 or not np.isfinite(voxel_size) or any(int(s) != s or s < 1 for s in shape):
        raise ValueError("voxel_size and grid_shape must be positive")
    shape = tuple(int(s) for s in shape)
    C = features.shape[1]
    cell = np.floor((points - origin) / voxel_size)  # [N,3] float cell index
    inside = ((cell >= 0) & (cell < np.array(shape))).all(axis=1)  # before any int cast
    index = cell[inside].astype(np.int64)
    if len(index) == 0:
        return np.zeros((0, 3), np.int64), np.zeros((0, C)), np.zeros(0, np.int64)
    flat = np.ravel_multi_index(index.T, shape)  # C order == lexicographic x,y,z
    unique, inverse, counts = np.unique(flat, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    sums = np.zeros((len(unique), C))
    np.add.at(sums, inverse, features[inside])
    coords = np.stack(np.unravel_index(unique, shape), axis=1).astype(np.int64)
    return coords, sums / counts[:, None], counts.astype(np.int64)


# %% [verifiers]
def verify_baseline_worked_example(voxel_mean):
    points = np.array([[0.1, 0, 0], [0.9, 0, 0], [1.0, 0, 0], [2.0, 0, 0]])
    features = np.array([[2.], [4.], [8.], [99.]])
    coords, means, counts = voxel_mean(points, features, (0, 0, 0), 1.0, (2, 1, 1))
    assert np.array_equal(coords, [[0, 0, 0], [1, 0, 0]])
    assert np.allclose(means, [[3.], [8.]]) and np.array_equal(counts, [2, 1])


def verify_baseline_empty_inputs_have_correct_shapes(voxel_mean):
    coords, means, counts = voxel_mean(np.zeros((0, 3)), np.zeros((0, 4)), np.zeros(3), 0.5, (2, 2, 2))
    assert coords.shape == (0, 3) and means.shape == (0, 4) and counts.shape == (0,)
    coords, means, counts = voxel_mean(np.array([[9., 9., 9.]]), np.ones((1, 2)), np.zeros(3), 1.0, (2, 2, 2))
    assert coords.shape == (0, 3) and means.shape == (0, 2) and counts.shape == (0,)


def verify_followup_origin_negative_points_and_boundaries(voxel_mean):
    pts = np.array([[-0.1, 0, 0], [-1.0, 0, 0], [0.0, 0, 0], [1.99, 0, 0], [2.0, 0, 0]])
    feats = np.arange(5.).reshape(5, 1)
    coords, means, counts = voxel_mean(pts, feats, (-1.0, 0, 0), 1.0, (3, 1, 1))
    # (x - origin): 0.9, 0.0, 1.0, 2.99, 3.0 -> cells 0, 0, 1, 2, 3(out)
    assert np.array_equal(coords[:, 0], [0, 1, 2]) and np.array_equal(counts, [2, 1, 1])
    assert np.allclose(means[:, 0], [0.5, 2.0, 3.0])


def verify_followup_lexicographic_order_and_multichannel_means(voxel_mean):
    pts = np.array([[1.5, 0.5, 0.5], [0.5, 1.5, 0.5], [0.5, 0.5, 1.5], [0.5, 0.5, 0.5], [0.6, 0.4, 0.4]])
    feats = np.array([[1., 10.], [2., 20.], [3., 30.], [4., 40.], [6., 60.]])
    coords, means, counts = voxel_mean(pts, feats, np.zeros(3), 1.0, (2, 2, 2))
    keys = [tuple(c) for c in coords]
    assert keys == sorted(keys) and keys[0] == (0, 0, 0)
    assert counts[0] == 2 and np.allclose(means[0], [5., 50.])


def verify_followup_matches_dictionary_reference(voxel_mean):
    rng = np.random.default_rng(0)
    pts, feats = rng.uniform(-1, 5, size=(300, 3)), rng.normal(size=(300, 3))
    origin, size, shape = np.array([0.5, -0.2, 0.1]), 0.7, (5, 4, 6)
    coords, means, counts = voxel_mean(pts, feats, origin, size, shape)
    table = {}
    for p, f in zip(pts, feats):
        c = tuple(np.floor((p - origin) / size).astype(int))
        if all(0 <= c[a] < shape[a] for a in range(3)):
            table.setdefault(c, []).append(f)
    assert [tuple(c) for c in coords] == sorted(table)
    for c, m, n in zip(coords, means, counts):
        assert n == len(table[tuple(c)]) and np.allclose(m, np.mean(table[tuple(c)], axis=0))


def verify_production_huge_coordinates_and_invalid_contracts(voxel_mean):
    pts = np.array([[1e300, 0, 0], [-1e300, 0, 0], [0.5, 0.5, 0.5]])
    coords, means, counts = voxel_mean(pts, np.array([[1.], [2.], [3.]]), np.zeros(3), 1.0, (1, 1, 1))
    assert np.array_equal(coords, [[0, 0, 0]]) and counts[0] == 1 and means[0, 0] == 3.0
    ok = (np.zeros((1, 3)), np.zeros((1, 1)), np.zeros(3), 1.0, (1, 1, 1))
    for i, bad in ((0, np.array([[np.nan, 0, 0]])), (1, np.array([[np.inf]])), (3, 0.0), (3, -1.0),
                   (4, (0, 1, 1)), (4, (1, 1)), (1, np.zeros((2, 1)))):
        args = list(ok); args[i] = bad
        try:
            voxel_mean(*args)
        except ValueError:
            continue
        raise AssertionError(f"invalid argument {i} accepted")


def verify_production_pure_and_vectorized(voxel_mean):
    import time
    rng = np.random.default_rng(1)
    pts, feats = rng.uniform(0, 100, size=(200_000, 3)), rng.normal(size=(200_000, 4))
    p0, f0 = pts.copy(), feats.copy()
    start = time.perf_counter()
    coords, means, counts = voxel_mean(pts, feats, np.zeros(3), 1.0, (100, 100, 100))
    assert time.perf_counter() - start < 5.0, "a Python loop over points is too slow"
    assert counts.sum() == 200_000 and means.shape == (len(coords), 4)
    assert np.array_equal(pts, p0) and np.array_equal(feats, f0)


# %%
from torch_judge import evaluate
if not evaluate("0130_voxel_feature_mean_eval.py", voxel_mean):
    raise SystemExit(1)
