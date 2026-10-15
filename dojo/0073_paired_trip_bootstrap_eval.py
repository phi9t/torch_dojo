"""Three-stage reference for paired trip-level percentile bootstrap."""

# %%
import numpy as np
from hypothesis import given, settings, strategies as st

# %%
# Drill 1: Paired resampling prints `[1. 1.]`. Every trip's difference is
# exactly 1, so any resample of differences averages to 1 and the interval has
# zero width. Independent resampling prints roughly `[-4.333  6.333]`: the
# baseline and candidate draws pick different trips, so the spread between
# trips (losses 0 to 9) floods the interval, which now even includes "the
# candidate is worse". Pairing cancels that shared trip-to-trip variation.
#
# Drill 2: It prints `[0.1 3.9]`, and neither value is in the array. With
# linear interpolation the $q$-th percentile of $n$ sorted values sits at
# position $(n-1)q/100$: $4 \times 0.025 = 0.1$ lies between 0 and 1, and
# $4 \times 0.975 = 3.9$ between 3 and 4. Picking the nearest observed value
# instead would give a different interval than the contract asks for.
#
# Drill 3: The trip mean prints `5.0`; the frame mean prints `1.0`. Pooling
# frames lets the long trip B outvote trip A nine to one. The bootstrap treats
# trips as the independent units, so each trip's loss counts once, however
# many frames it has.

# %%
def paired_trip_bootstrap(baseline: np.ndarray, candidate: np.ndarray,
                          iterations: int, rng: np.random.Generator
                          ) -> tuple[float, tuple[float, float]]:
    baseline = np.asarray(baseline)
    candidate = np.asarray(candidate)
    if (baseline.ndim != 1 or candidate.shape != baseline.shape
            or len(baseline) < 2 or baseline.dtype.kind not in 'fi'
            or candidate.dtype.kind not in 'fi'
            or not np.isfinite(baseline).all() or not np.isfinite(candidate).all()
            or isinstance(iterations, (bool, np.bool_))
            or not isinstance(iterations, (int, np.integer)) or iterations < 1):
        raise ValueError('need finite paired trip losses and positive iterations')
    differences = baseline.astype(np.float64) - candidate.astype(np.float64)
    if not np.isfinite(differences).all():
        raise ValueError('trip differences overflowed')
    means = np.empty(iterations, dtype=np.float64)
    for j in range(iterations):
        indices = rng.integers(0, len(differences), size=len(differences))
        means[j] = differences[indices].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(differences.mean()), (float(lo), float(hi))


# %% [verifiers]
def verify_baseline_constant_improvement(paired_trip_bootstrap):
    result = paired_trip_bootstrap(np.array([2., 3., 4.]), np.array([1., 2., 3.]),
                                   20, np.random.default_rng(0))
    assert result == (1., (1., 1.))


def verify_baseline_point_estimate_and_sign(paired_trip_bootstrap):
    a, b = np.array([1., 10., 2.]), np.array([2., 8., 4.])
    estimate, ci = paired_trip_bootstrap(a, b, 1000, np.random.default_rng(4))
    assert np.isclose(estimate, -1/3)
    assert ci[0] <= estimate <= ci[1]


def verify_followup_manual_paired_replay(paired_trip_bootstrap):
    a, b = np.array([1., 8., 11.]), np.array([3., 3., 5.])
    seed = 17
    draws = np.random.default_rng(seed).integers(0, 3, size=(7, 3))
    expected_means = (a-b)[draws].mean(axis=1)
    expected = (float((a-b).mean()), tuple(np.percentile(expected_means, [2.5, 97.5])))
    got = paired_trip_bootstrap(a, b, 7, np.random.default_rng(seed))
    replay = paired_trip_bootstrap(a, b, 7, np.random.default_rng(seed))
    assert got == replay
    assert np.allclose(got[0], expected[0]) and np.allclose(got[1], expected[1])


def verify_followup_shift_both_models(paired_trip_bootstrap):
    a, b = np.array([2., 4., 8.]), np.array([3., 1., 2.])
    initial = paired_trip_bootstrap(a, b, 29, np.random.default_rng(23))
    shifted = paired_trip_bootstrap(a+100, b+100, 29, np.random.default_rng(23))
    assert np.allclose(initial[0], shifted[0]) and np.allclose(initial[1], shifted[1])


def verify_production_invalid_and_immutable(paired_trip_bootstrap):
    a = np.array([2., 3.]); b = np.array([1., 1.])
    before = a.copy()
    paired_trip_bootstrap(a, b, 2, np.random.default_rng(1))
    assert np.array_equal(a, before) and np.array_equal(b, [1., 1.])
    invalid = [
        (np.array([1.]), np.array([0.]), 2),
        (a, np.array([1.]), 2),
        (np.array([[1., 2.]]), np.array([[0., 0.]]), 2),
        (np.array([np.nan, 2.]), b, 2),
        (a, b, 0),
        (a, b, 1.5),
    ]
    for base, cand, count in invalid:
        try:
            paired_trip_bootstrap(base, cand, count, np.random.default_rng(0))
        except ValueError:
            pass
        else:
            raise AssertionError('invalid bootstrap input accepted')


def verify_production_generated_constant_gap(paired_trip_bootstrap):
    @settings(max_examples=20, deadline=None, derandomize=True)
    @given(st.lists(st.floats(-100, 100, allow_nan=False, allow_infinity=False),
                    min_size=2, max_size=20))
    def check(values):
        baseline = np.asarray(values)
        candidate = baseline - 2.
        estimate, ci = paired_trip_bootstrap(baseline, candidate, 17,
                                             np.random.default_rng(3))
        assert np.isclose(estimate, 2., atol=1e-12)
        assert np.allclose(ci, (2., 2.), atol=1e-12)
    check()


# %%
from torch_judge import evaluate
if not evaluate('0073_paired_trip_bootstrap_eval.py', paired_trip_bootstrap):
    raise SystemExit(1)
