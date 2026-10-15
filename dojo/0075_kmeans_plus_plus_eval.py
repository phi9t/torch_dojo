"""Staged reference and verifiers for squared-distance centroid seeding."""

# %%
import numpy as np
from hypothesis import given, settings, strategies as st

# Drill 1: Distances print `[0.  0.1 0.9]`; squared distances print
# `[0.    0.012 0.988]`. Squaring is what makes K-means++ favour far points so
# strongly: point 9 is 9 times farther than point 1 but 81 times as likely.
# Plain distance weights are a different, weaker seeding rule.

# Drill 2: The first line prints `[9 6 6] [9 6 6]`: two fresh Generators with
# the same seed replay the same stream. The second prints `[9 6 6] [8 5 7]`:
# one Generator advances its state on every draw. So use only the `rng` the
# caller passes, never the global `np.random` state or a new seed inside the
# function; replay comes from the caller handing in a freshly seeded Generator.

# Drill 3: Latest-only prints `[100.  81.   1.   0.]`, nearest prints
# `[0. 1. 1. 0.]`. Point 1 sits right next to center 0, so its weight is 1,
# not 81, and the already chosen point 0 would get weight 100 if you forgot
# center 0. The weight is the distance to the closest chosen center, kept as a
# running `np.minimum` over every center so far.

# %%
def kmeans_plus_plus(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    raw = np.asarray(X)
    if (raw.ndim != 2 or raw.shape[1] == 0 or raw.dtype.kind not in "iuf"
            or isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer))
            or not 1 <= k <= len(raw) or not isinstance(rng, np.random.Generator)):
        raise ValueError("need finite [N,D] points and 1 <= k <= N")
    X = raw.astype(float)
    if not np.isfinite(X).all():
        raise ValueError("need finite [N,D] points and 1 <= k <= N")
    chosen = [int(rng.integers(len(X)))]
    used = np.zeros(len(X), dtype=bool)
    used[chosen[0]] = True
    with np.errstate(over="ignore", invalid="ignore"):
        nearest_d2 = np.sum((X - X[chosen[0]]) ** 2, axis=1)
    while len(chosen) < k:
        nearest_d2[used] = 0
        maximum = float(nearest_d2.max())
        if not np.isfinite(maximum):
            raise ValueError("squared-distance overflow")
        if maximum == 0:
            index = int(rng.choice(np.flatnonzero(~used)))
        else:
            scaled = nearest_d2 / maximum
            index = int(rng.choice(len(X), p=scaled / scaled.sum()))
        chosen.append(index)
        used[index] = True
        if len(chosen) < k:
            with np.errstate(over="ignore", invalid="ignore"):
                new_distance = np.sum((X - X[index]) ** 2, axis=1)
            nearest_d2 = np.minimum(nearest_d2, new_distance)
    return X[chosen].copy()


# %% [verifiers]
def verify_followup_all_identical_points_and_distinct_indices(kmeans_plus_plus):
    X = np.ones((4, 2))
    result = kmeans_plus_plus(X, 4, np.random.default_rng(0))
    assert result.shape == (4, 2) and np.array_equal(result, X)


def verify_production_seeded_replay_and_no_mutation(kmeans_plus_plus):
    X = np.array([[0., 0.], [1., 1.], [4., 0.], [10., 2.]])
    original = X.copy()
    a = kmeans_plus_plus(X, 3, np.random.default_rng(7))
    b = kmeans_plus_plus(X, 3, np.random.default_rng(7))
    assert np.array_equal(a, b) and np.array_equal(X, original)
    assert len({tuple(row) for row in a}) == 3
    a[0] = -100
    assert np.array_equal(X, original), "return a copy of the selected rows"


def verify_baseline_all_rows_when_k_equals_n(kmeans_plus_plus):
    X = np.arange(15., dtype=float).reshape(5, 3)
    result = kmeans_plus_plus(X, len(X), np.random.default_rng(2))
    assert sorted(map(tuple, result)) == sorted(map(tuple, X))


def verify_production_reject_invalid_k_and_points(kmeans_plus_plus):
    for X, k in [(np.zeros((2, 1)), 0), (np.zeros((2, 1)), 3), (np.array([[np.inf]]), 1)]:
        try:
            kmeans_plus_plus(X, k, np.random.default_rng(0))
        except ValueError:
            pass
        else:
            raise AssertionError("invalid input accepted")


def verify_baseline_weighted_choice(kmeans_plus_plus):
    X = np.array([[0.], [1.], [9.]])
    result = kmeans_plus_plus(X, 2, np.random.default_rng(0))
    assert result.shape == (2, 1)
    assert result[0, 0] != result[1, 0]


def verify_followup_k_one_and_duplicate_coordinates(kmeans_plus_plus):
    X = np.array([[2., 3.], [2., 3.], [2., 3.]])
    assert np.array_equal(kmeans_plus_plus(X, 1, np.random.default_rng(1)), [[2., 3.]])
    assert np.array_equal(kmeans_plus_plus(X, 3, np.random.default_rng(1)), X)


def verify_production_reject_boolean_k_and_bad_rng(kmeans_plus_plus):
    X = np.array([[0.], [1.]])
    for k, rng in ((True, np.random.default_rng(0)), (1, None)):
        try:
            kmeans_plus_plus(X, k, rng)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid k or RNG accepted")


def verify_production_weight_normalization(kmeans_plus_plus):
    # Both remaining squared distances are finite; their raw sum overflows.
    X = np.array([[-1e154], [0.], [1e154]])
    result = kmeans_plus_plus(X, 2, np.random.default_rng(1))
    assert result[0, 0] == 0.0
    assert abs(result[1, 0]) == 1e154


def verify_production_generated_seeded_replay(kmeans_plus_plus):
    @settings(max_examples=20, derandomize=True, deadline=None)
    @given(st.lists(st.integers(-20, 20), min_size=1, max_size=12), st.integers(0, 1000))
    def property_check(values, seed):
        X = np.asarray(values, dtype=float)[:, None]
        original = X.copy()
        a = kmeans_plus_plus(X, len(X), np.random.default_rng(seed))
        b = kmeans_plus_plus(X, len(X), np.random.default_rng(seed))
        assert np.array_equal(a, b)
        assert sorted(a[:, 0].tolist()) == sorted(X[:, 0].tolist())
        assert np.array_equal(X, original)
    property_check()


# %%
from torch_judge import evaluate
if not evaluate("0075_kmeans_plus_plus_eval.py", kmeans_plus_plus):
    raise SystemExit(1)
