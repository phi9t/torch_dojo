"""Three-stage reference for multiclass calibration metrics."""

# %%
import numpy as np
from hypothesis import given, settings, strategies as st

# %%
# Drill 1: The floor rule prints `[0 1 1]` and the ceil-minus-one rule prints
# `[0 0 1]`; they disagree only on 0.5, which sits exactly on the interior edge.
# The contract's bins are $[0, 0.5]$ and $(0.5, 1]$, so 0.5 belongs to bin 0.
# `floor(0.5 * 2) = 1` sends an edge to the bin on its right, while
# `ceil(c * B) - 1` puts each edge at the top of the bin on its left. The floor
# rule also sends confidence 1 past the last bin, so it needs a clamp there; the
# ceil rule needs one only at confidence 0.
#
# Drill 2: The squared errors are `[.04, .04]` for row 0 and `[.36, .36]` for
# row 1. Summing within each row gives .08 and .72, whose mean is `0.4`, the
# multiclass Brier score. `squared.mean()` divides by all $NC = 4$ entries and
# prints `0.2`: the same quantity silently scaled by $1/C$.
#
# Drill 3: It prints `[0.8 0.6]`, `[0.8 0.4]` and `[ True False]`. When the
# prediction is right, the argmax is the true class, so the two numbers agree.
# When it is wrong they differ: the model claimed class 0 with confidence .6.
# Calibration asks whether predictions made at 60% are right 60% of the time,
# so ECE uses the max (.6). NLL scores the true class (.4), and Brier scores
# every class against the one-hot target.

# %%
def calibration_metrics(probabilities: np.ndarray, labels: np.ndarray,
                        bins: int = 10) -> tuple[float, float, float]:
    p = np.asarray(probabilities)
    y = np.asarray(labels)
    if (p.ndim != 2 or p.shape[0] < 1 or p.shape[1] < 2
            or p.dtype.kind not in 'f' or not np.isfinite(p).all()
            or np.any((p < 0) | (p > 1))
            or not np.allclose(p.sum(axis=1), 1, rtol=0, atol=1e-10)
            or y.shape != (len(p),) or y.dtype.kind not in 'iu'
            or np.any((y < 0) | (y >= p.shape[1]))
            or isinstance(bins, (bool, np.bool_))
            or not isinstance(bins, (int, np.integer)) or bins < 1):
        raise ValueError('invalid probabilities, labels, or bins')
    confidence = p.max(axis=1)
    correct = p.argmax(axis=1) == y
    # Exact interior edges fall into the bin to their left.
    bucket = np.clip(np.ceil(confidence * bins).astype(int) - 1, 0, bins - 1)
    ece = 0.0
    for index in range(bins):
        selected = bucket == index
        if selected.any():
            ece += selected.mean() * abs(correct[selected].mean() - confidence[selected].mean())
    one_hot = np.eye(p.shape[1])[y]
    brier = np.square(p - one_hot).sum(axis=1).mean()
    nll = -np.log(np.maximum(p[np.arange(len(p)), y], np.finfo(np.float64).tiny)).mean()
    return float(ece), float(brier), float(nll)


# %% [verifiers]
def verify_baseline_perfect_and_wrong(calibration_metrics):
    exact = calibration_metrics(np.array([[1., 0.], [0., 1.]]), np.array([0, 1]))
    wrong = calibration_metrics(np.array([[1., 0.]]), np.array([1]))
    assert np.allclose(exact, 0)
    assert np.isclose(wrong[0], 1) and np.isclose(wrong[1], 2)
    assert np.isfinite(wrong[2])


def verify_baseline_known_denominators(calibration_metrics):
    p = np.array([[.8, .2], [.6, .4]])
    got = calibration_metrics(p, np.array([0, 1]), bins=2)
    expected = (.2, .4, -(np.log(.8) + np.log(.4))/2)
    assert np.allclose(got, expected)


def verify_followup_bin_edges_and_empty_bins(calibration_metrics):
    p = np.array([[.5, .5], [.75, .25], [1., 0.]])
    # .5 occupies first bin, .75 and 1 occupy second; all predictions are class 0.
    ece, _, _ = calibration_metrics(p, np.array([0, 1, 0]), bins=2)
    expected = (abs(1-.5) + 2*abs(.5-.875))/3
    assert np.isclose(ece, expected)
    assert np.isclose(calibration_metrics(np.array([[1., 0.]]), np.array([0]), bins=20)[0], 0)


def verify_followup_argmax_tie_and_classwise_brier(calibration_metrics):
    ece, brier, nll = calibration_metrics(np.array([[.5, .5, 0.]]), np.array([1]), bins=4)
    assert np.isclose(ece, .5) and np.isclose(brier, .5) and np.isclose(nll, np.log(2))


def verify_production_validation_and_immutability(calibration_metrics):
    p = np.array([[.3, .7]])
    y = np.array([1])
    before = p.copy()
    calibration_metrics(p, y)
    assert np.array_equal(p, before) and np.array_equal(y, [1])
    invalid = [
        (np.array([[.2, .2]]), y, 2),
        (np.array([[np.nan, 1.]]), y, 2),
        (np.array([[1.2, -.2]]), y, 2),
        (p, np.array([-1]), 2),
        (p, np.array([2]), 2),
        (p, np.array([1.]), 2),
        (p, y, 0),
        (p, y, 1.5),
        (np.empty((0, 2)), np.empty(0, dtype=int), 2),
    ]
    for bad_p, bad_y, bad_bins in invalid:
        try:
            calibration_metrics(bad_p, bad_y, bad_bins)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid calibration input accepted')


def verify_production_generated_one_hot(calibration_metrics):
    @settings(max_examples=20, deadline=None, derandomize=True)
    @given(st.lists(st.integers(0, 3), min_size=1, max_size=15))
    def check(labels):
        y = np.array(labels)
        p = np.eye(4)[y]
        assert np.allclose(calibration_metrics(p, y, bins=7), 0)
    check()


# %%
from torch_judge import evaluate
if not evaluate('0069_calibration_metrics_eval.py', calibration_metrics):
    raise SystemExit(1)
