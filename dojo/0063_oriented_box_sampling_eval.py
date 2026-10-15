"""Reference and cases for Uniform Oriented-Box Sampling."""

# %%
import numpy as np

# %%
# Drill 1: It prints `[[0. 1. 0.]] [[ 0. -1.  0.]]`. The rotation matrix is
# written for column vectors, $p_{world} = R\,p_{local}$. Transposing both sides
# gives $p_{world}^T = p_{local}^T R^T$, so a row-stored point needs `@ R.T`,
# which sends box-local $+x$ to world $+y$ as a counter-clockwise yaw should.
# `local @ R` computes $R^T p$, the INVERSE rotation by $-\text{yaw}$, so the
# car faces world $-y$. At yaw 0 both agree, which is why this bug hides.
#
# Drill 2: `np.cos` and `np.sin` read radians. A yaw of `90` is 90 radians,
# about 14.3 turns, which leaves $90 \bmod 2\pi \approx 2.04$ rad
# ($\approx 117^\circ$), so it prints `[-0.448  0.894]` instead of `[0. 1.]`.
# Nothing raises: every yaw is a valid angle, so a degrees-for-radians mix-up
# silently samples a box turned the wrong way. Convert with `np.deg2rad` at the
# boundary.
#
# Drill 3: It prints `[[2. 0. 0.]] [[0. 2. 0.]]`. Yaw rotates the box about its
# own centre, so the box-local origin must stay at the centre: rotate in the
# box-local frame first, then add `center`. Adding the centre first moves the
# point into world coordinates, and the rotation then swings it about the WORLD
# origin, carrying the whole car from $[2, 0, 0]$ to $[0, 2, 0]$.

# %%
def sample_oriented_box(center: np.ndarray, size: np.ndarray, yaw: float, count: int, rng: np.random.Generator) -> np.ndarray:
    center, size = np.asarray(center), np.asarray(size)
    if (center.shape != (3,) or size.shape != (3,)
            or center.dtype.kind not in 'fiu' or size.dtype.kind not in 'fiu'
            or not np.isfinite(center).all() or not np.isfinite(size).all()
            or not np.all(size > 0)
            or not isinstance(yaw, (int, float, np.integer, np.floating))
            or isinstance(yaw, (bool, np.bool_))
            or not np.isfinite(yaw)
            or not isinstance(count, (int, np.integer)) or isinstance(count, (bool, np.bool_))
            or count < 0 or not isinstance(rng, np.random.Generator)):
        raise ValueError("invalid box or count")
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    local = rng.uniform(-.5, .5, size=(count, 3)) * size
    return local @ R.T + center


# %% [verifiers]
def verify_baseline_axis_aligned_probe(sample_oriented_box):
    center = np.array([2., -1., 3.])
    size = np.array([2., 4., 6.])
    seed = 9
    expected = np.random.default_rng(seed).uniform(-.5, .5, size=(4, 3)) * size + center
    actual = sample_oriented_box(center, size, 0., 4, np.random.default_rng(seed))
    assert actual.shape == (4, 3) and np.allclose(actual, expected)


def verify_followup_bounds_and_replay(sample_oriented_box):
    center = np.array([2.0, 3.0, 4.0])
    size = np.array([2.0, 4.0, 6.0])
    yaw = np.pi / 3
    a = sample_oriented_box(center, size, yaw, 1000, np.random.default_rng(4))
    b = sample_oriented_box(center, size, yaw, 1000, np.random.default_rng(4))
    c, s = (np.cos(yaw), np.sin(yaw))
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    local = (a - center) @ R
    assert np.array_equal(a, b) and np.all(np.abs(local) <= size / 2 + 1e-12)


def verify_followup_uniform_moments(sample_oriented_box):
    size = np.array([2.0, 4.0, 6.0])
    a = sample_oriented_box(np.zeros(3), size, 0.0, 50000, np.random.default_rng(1))
    assert np.allclose(a.mean(0), 0, atol=0.05) and np.allclose(a.var(0), size ** 2 / 12, rtol=0.04)


def verify_production_zero_samples_and_invalid_size(sample_oriented_box):
    assert sample_oriented_box(np.zeros(3), np.ones(3), 0.0, 0, np.random.default_rng(0)).shape == (0, 3)
    try:
        sample_oriented_box(np.zeros(3), np.array([1.0, 0.0, 1.0]), 0.0, 1, np.random.default_rng(0))
    except ValueError:
        pass
    else:
        raise AssertionError('zero size accepted')


def verify_production_integer_count_and_inputs(sample_oriented_box):
    center, size = np.zeros(3), np.ones(3)
    for bad_count in (-1, 1.5, True):
        try:
            sample_oriented_box(center, size, 0., bad_count, np.random.default_rng(0))
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid count accepted: {bad_count!r}")
    try:
        sample_oriented_box(center, size, 0., 1, None)
    except ValueError:
        pass
    else:
        raise AssertionError("missing injected Generator accepted")
    for bad_center, bad_size, bad_yaw in (
        (np.array([np.nan, 0., 0.]), size, 0.),
        (center, np.array([1., -1., 1.]), 0.),
        (center, size, np.inf),
        (center, size, "north"),
        (center, size, 1j),
        (center, size, True),
    ):
        try:
            sample_oriented_box(bad_center, bad_size, bad_yaw, 1, np.random.default_rng(0))
        except ValueError:
            pass
        else:
            raise AssertionError("invalid geometry accepted")


def verify_production_immutability_and_large_batch(sample_oriented_box):
    center, size = np.array([3., 4., 5.]), np.array([2., 3., 4.])
    original_center, original_size = center.copy(), size.copy()
    result = sample_oriented_box(center, size, -.7, 50001, np.random.default_rng(5))
    assert result.shape == (50001, 3) and np.isfinite(result).all()
    assert np.array_equal(center, original_center) and np.array_equal(size, original_size)


# %%
from torch_judge import evaluate
if not evaluate("0063_oriented_box_sampling_eval.py", sample_oriented_box):
    raise SystemExit(1)
