"""Reference and checks for fitted binary probability calibration."""

# %%
import numpy as np

# Drill 1: The sigmoid prints `[0.047 0.5   0.953]`; applied again it prints
# `[0.512 0.622 0.722]`. The sigmoid of anything in $[0, 1]$ lies in
# $[0.5, 0.731]$, so a squashed input can never map to a low probability.
# Platt scaling is a line in logit space: fit $a s + b$ on the raw scores and
# apply the sigmoid exactly once, to that line, on the way out.

# Drill 2: The plain loss prints `0.313262`, `4.5e-05`, `0.0` and keeps
# falling as the slope grows, because separable data has no finite
# maximum-likelihood slope. With the penalty it prints `0.313762`, `0.050045`,
# `5.0`: the $a^2$ term grows faster than the loss shrinks, so the fit has a
# finite optimum and the probabilities stay off exactly 0 and 1.

# Drill 3: Labels `[0, 1, 0]` give `[0.0, 0.5, 0.5]` and `[1, 0, 0]` give
# `[0.333, 0.333, 0.333]`. Rows with the same score would get different
# levels depending on an arbitrary tie order. Averaging identical scores into
# one weighted knot first (here $1/2$ with weight 2) makes the fit independent
# of row order and gives score 0 a single level.

# %%
def calibrate_binary(scores, labels, query_scores, method='platt'):
    raw_x = np.asarray(scores)
    raw_y = np.asarray(labels)
    raw_q = np.asarray(query_scores)
    if (raw_x.ndim != 1 or len(raw_x) < 1 or raw_x.dtype.kind not in 'fiu'
            or raw_y.shape != raw_x.shape or raw_y.dtype.kind not in 'iu'
            or raw_q.ndim != 1 or raw_q.dtype.kind not in 'fiu'
            or not np.isfinite(raw_x).all() or not np.isfinite(raw_q).all()
            or not np.all((raw_y == 0) | (raw_y == 1))
            or method not in ('platt', 'isotonic')):
        raise ValueError('need finite 1-D scores, binary integer labels, and a known method')
    x = raw_x.astype(float)
    y = raw_y.astype(float)
    q = raw_q.astype(float)

    def sigmoid(z):
        positive = z >= 0
        result = np.empty_like(z, dtype=float)
        result[positive] = 1 / (1 + np.exp(-z[positive]))
        e = np.exp(z[~positive])
        result[~positive] = e / (1 + e)
        return result

    if method == 'isotonic':
        knots, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
        sums = np.bincount(inverse, weights=y, minlength=len(knots))
        # Each block stores its first and final knot, total label, and weight.
        blocks = []
        for index, (total, count) in enumerate(zip(sums, counts)):
            blocks.append([index, index, total, count])
            while (len(blocks) > 1 and
                   blocks[-2][2] / blocks[-2][3] > blocks[-1][2] / blocks[-1][3]):
                right = blocks.pop()
                left = blocks[-1]
                left[1] = right[1]
                left[2] += right[2]
                left[3] += right[3]
        levels = np.empty(len(knots))
        for first, last, total, count in blocks:
            levels[first:last+1] = total / count
        return np.interp(q, knots, levels)

    # Fit the slope against bounded features, but penalize its raw-score value.
    inverse_scale = 1 / max(1., np.max(np.abs(x)))
    feature = x * inverse_scale
    query_feature = q * inverse_scale
    slope_penalty = .001 * inverse_scale * inverse_scale

    def objective(a, b):
        z = a * feature + b
        return np.logaddexp(0, (1 - 2*y) * z).mean() + .0005 * ((a*inverse_scale)**2 + b*b)

    a = b = 0.
    for _ in range(100):
        p = sigmoid(a*feature + b)
        residual = p - y
        gradient = np.array([np.mean(residual*feature) + slope_penalty*a,
                             np.mean(residual) + .001*b])
        if np.max(np.abs(gradient)) < 1e-12:
            break
        weight = p * (1-p)
        hessian = np.array([[np.mean(weight*feature*feature) + slope_penalty,
                             np.mean(weight*feature)],
                            [np.mean(weight*feature), np.mean(weight) + .001]])
        step = np.linalg.solve(hessian, gradient)
        scale = 1.
        old = objective(a, b)
        while scale > 1e-12 and objective(a-scale*step[0], b-scale*step[1]) > old:
            scale *= .5
        a -= scale * step[0]
        b -= scale * step[1]
    return sigmoid(a*query_feature + b)


# %% [verifiers]
def verify_baseline_platt_known_symmetric_optimum(calibrate_binary):
    # For (-1,0),(1,1), symmetry gives b=0 and the regularized
    # first-order condition sigmoid(a) - 1 + .001*a = 0.
    lo, hi = 0., 20.
    for _ in range(70):
        mid = (lo + hi) / 2
        if 1 / (1 + np.exp(-mid)) - 1 + .001 * mid < 0:
            lo = mid
        else:
            hi = mid
    slope = (lo + hi) / 2
    expected = 1 / (1 + np.exp(-np.array([-slope, 0., slope])))
    got = calibrate_binary(np.array([-1., 1.]), np.array([0, 1]),
                           np.array([-1., 0., 1.]))
    assert got.shape == (3,) and np.allclose(got, expected, atol=1e-7)


def verify_baseline_platt_penalizes_intercept(calibrate_binary):
    # With zero scores, a=0. The intercept solves sigmoid(b)-.75+.001*b=0.
    lo, hi = 0., 10.
    for _ in range(70):
        mid = (lo + hi) / 2
        if 1 / (1 + np.exp(-mid)) - .75 + .001 * mid < 0:
            lo = mid
        else:
            hi = mid
    expected = 1 / (1 + np.exp(-(lo + hi) / 2))
    got = calibrate_binary(np.zeros(4), np.array([0, 1, 1, 1]),
                           np.array([-4., 0., 4.]))
    assert np.allclose(got, expected, atol=1e-8)


def verify_followup_isotonic_ties_and_interpolation(calibrate_binary):
    # At score 0, labels average to 1/2 before PAV. The score-1 mean
    # is zero, so their weighted pooled level is (1+0+0)/3 = 1/3.
    got = calibrate_binary(np.array([0., 0., 1., 2.]),
                           np.array([0, 1, 0, 1]),
                           np.array([-2., 0., .5, 1., 1.5, 2., 5.]),
                           method='isotonic')
    assert np.allclose(got, [1/3, 1/3, 1/3, 1/3, 2/3, 1., 1.])
    assert np.all(np.diff(got) >= 0)


def verify_production_constant_one_class_and_empty_queries(calibrate_binary):
    for method in ('platt', 'isotonic'):
        got = calibrate_binary(np.zeros(4), np.array([0, 1, 1, 1]),
                               np.array([-10., 0., 10.]), method)
        assert got.shape == (3,) and np.isfinite(got).all()
        assert np.all((got >= 0) & (got <= 1))
        assert np.allclose(got, got[0])
        if method == 'isotonic':
            assert np.allclose(got, .75)
        else:
            assert abs(got[0] - .75) < .01
        one_class = calibrate_binary(np.array([-2., 0., 2.]), np.ones(3, dtype=int),
                                     np.array([-1., 1.]), method)
        assert np.isfinite(one_class).all() and np.all((one_class >= 0) & (one_class <= 1))
        assert calibrate_binary(np.array([0.]), np.array([1]), np.array([]), method).shape == (0,)


def verify_production_validation_immutability_and_heldout(calibrate_binary):
    scores = np.array([-2., -1., 1., 2.])
    labels = np.array([0, 0, 1, 1])
    query = np.array([-1.5, .0, 1.5])
    originals = [array.copy() for array in (scores, labels, query)]
    for method in ('platt', 'isotonic'):
        probabilities = calibrate_binary(scores, labels, query, method)
        assert probabilities.shape == query.shape
        assert np.isfinite(probabilities).all()
        assert np.all((probabilities >= 0) & (probabilities <= 1))
        assert probabilities[0] < probabilities[1] < probabilities[2]
        # Held-out outcomes appear only in this verifier, never in fitting.
        heldout_labels = np.array([0, 0, 1])
        assert np.mean((probabilities - heldout_labels)**2) < .2
    for array, original in zip((scores, labels, query), originals):
        assert np.array_equal(array, original)
    invalid = [
        (np.array([]), np.array([], dtype=int), query, 'platt'),
        (scores[:, None], labels, query, 'platt'),
        (scores, labels[:2], query, 'platt'),
        (scores, labels.astype(float), query, 'platt'),
        (scores, np.array([0, 0, 1, 2]), query, 'platt'),
        (np.array([0., np.nan]), np.array([0, 1]), query, 'platt'),
        (scores, labels, np.array([np.inf]), 'platt'),
        (scores, labels, query[:, None], 'platt'),
        (scores, labels, query, 'unknown'),
    ]
    for args in invalid:
        try:
            calibrate_binary(*args)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid calibration input accepted')


def verify_production_platt_extreme_finite_scores(calibrate_binary):
    scores = np.array([-1e200, 1e200])
    labels = np.array([0, 1])
    probabilities = calibrate_binary(scores, labels, np.array([-1e200, 0., 1e200]))
    assert probabilities.shape == (3,) and np.isfinite(probabilities).all()
    assert probabilities[0] < .01
    assert abs(probabilities[1] - .5) < 1e-8
    assert probabilities[2] > .99


# %%
from torch_judge import evaluate
if not evaluate('0094_binary_calibration_eval.py', calibrate_binary):
    raise SystemExit(1)
