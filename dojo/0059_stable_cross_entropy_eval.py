"""Three-stage reference for stable softmax cross-entropy and manual gradients."""

# %%
import numpy as np
from hypothesis import given, settings, strategies as st

# %%
# Shape workout answers: paired indices have shape (N,), and broadcast row
# (N,1) / column (1,N) indices form an (N,N) Cartesian grid. For [[10,11,12],
# [20,21,22]] and targets [2,2], paired values are [12,22]; grid values are
# [[12,12],[22,22]]. Each row number is part of the address.
# normalizers[:,0] - selected is the desired per-example vector.
# normalizers - selected[:,None] also aligns rows, but returns (N,1).
# normalizers - selected returns (N,N) and mixes different examples.
#
# Derivation: L_i = logsumexp(z_i) - z_i[y_i]. Differentiating the first term
# gives exp(z_ij)/sum(exp(z_i)), and the second gives -1 when j=y_i, else 0.
# Therefore dL_i/dz_ij = p_ij - 1[j=y_i]. For a valid-row mean, divide by
# the valid count M; ignored rows have zero gradients. No dense one-hot
# array is needed: subtract 1 at the paired (row,label) coordinates.
#
# Drill 1: Two index arrays pair up element by element, so row $i$ meets
# column `targets[i]`: one value per example, `[12. 20.]`. A slice on the rows
# keeps every row and takes both target columns for each, a $2 \times 2$ grid
# `[[12. 10.] [22. 20.]]`. Only the paired form gives each example its own logit.
#
# Drill 2: $e^{1000}$ overflows float64 (anything above about $e^{709}$ is
# `inf`), so the naive form divides inf by inf and prints `[nan nan]`.
# Softmax is unchanged when the same constant is subtracted from every logit,
# so subtracting the row max keeps the answer, `[0.269 0.731]`, while the
# largest exponent becomes $e^0 = 1$.
#
# Drill 3: `mean()` divides by all 3 rows and prints `0.8333`. The ignored row
# adds zero loss but still inflates the denominator. Averaging over the 2 valid
# rows gives `1.25`, the loss the stage asks for.

# %%
def softmax_cross_entropy(logits: np.ndarray, targets: np.ndarray,
                          ignore_index: int = -100) -> tuple[float, np.ndarray]:
    logits = np.asarray(logits)
    targets = np.asarray(targets)
    if (logits.ndim != 2 or logits.shape[1] < 2 or logits.dtype != np.float64
            or not np.isfinite(logits).all()):
        raise ValueError("need finite float64 logits [N,C], C>=2")
    if targets.shape != (len(logits),) or targets.dtype.kind not in "iu" or not isinstance(ignore_index, (int, np.integer)):
        raise ValueError("need integer targets [N] and integer ignore_index")
    valid = targets != ignore_index
    if np.any((targets[valid] < 0) | (targets[valid] >= logits.shape[1])):
        raise ValueError("valid targets must be class IDs")
    gradient = np.zeros_like(logits)
    count = int(valid.sum())
    if count == 0:
        return 0.0, gradient
    selected = logits[valid]
    q = selected - selected.max(axis=1, keepdims=True)
    exponentials = np.exp(q)
    mass = exponentials.sum(axis=1, keepdims=True)
    probabilities = exponentials / mass
    row = np.arange(count)
    labels = targets[valid]
    loss = np.mean(np.log(mass[:, 0]) - q[row, labels])
    probabilities[row, labels] -= 1
    gradient[valid] = probabilities / count
    return float(loss), gradient


# %% [verifiers]
def verify_baseline_equal_logits_and_single_example(softmax_cross_entropy):
    loss, grad = softmax_cross_entropy(np.array([[0., 0.]]), np.array([1]))
    assert np.isclose(loss, np.log(2)) and np.allclose(grad, [[.5, -.5]])


def verify_baseline_mean_reduction(softmax_cross_entropy):
    z = np.array([[2., -1., 0.]])
    y = np.array([0])
    loss, grad = softmax_cross_entropy(z, y)
    repeated_loss, repeated_grad = softmax_cross_entropy(np.repeat(z, 3, axis=0), np.repeat(y, 3))
    assert np.isclose(loss, repeated_loss) and np.allclose(repeated_grad[0] * 3, grad[0])


def verify_baseline_repeated_labels_have_distinct_row_coordinates(softmax_cross_entropy):
    z = np.array([[0., 0., 0.], [0., np.log(2), 0.]])
    loss, grad = softmax_cross_entropy(z, np.array([1, 1]))
    assert np.isclose(loss, (np.log(3) + np.log(2)) / 2)
    assert grad.shape == z.shape
    assert np.allclose(grad, [[1/6, -1/3, 1/6], [1/8, -1/4, 1/8]])


def verify_production_asymmetric_per_row_losses_and_gradients(softmax_cross_entropy):
    # A wrong Cartesian subtraction has the SAME overall mean. Check rows
    # independently and the batched manual gradient, rather than only mean.
    z = np.array([[0., 0.], [0., 2.], [3., 1.]])
    y = np.array([1, 0, 1])
    expected_losses = np.array([np.log(2), np.log1p(np.exp(2)),
                               np.log1p(np.exp(2))])
    row_losses = np.array([softmax_cross_entropy(z[i:i+1], y[i:i+1])[0]
                           for i in range(3)])
    assert row_losses.shape == (3,) and np.allclose(row_losses, expected_losses)
    loss, grad = softmax_cross_entropy(z, y)
    p = 1 / (1 + np.exp(-2))
    assert np.isclose(loss, expected_losses.mean())
    assert grad.shape == z.shape
    assert np.allclose(grad, np.array([[.5, -.5], [-p, p], [p, -p]]) / 3)


def verify_followup_ignored_rows_and_all_ignored(softmax_cross_entropy):
    z = np.array([[1., 0.], [1000., -1000.], [0., 2.]])
    y = np.array([0, -100, 1])
    loss, grad = softmax_cross_entropy(z, y)
    expected_loss, expected_grad = softmax_cross_entropy(z[[0, 2]], np.array([0, 1]))
    assert np.isclose(loss, expected_loss) and np.allclose(grad[[0, 2]], expected_grad)
    assert np.array_equal(grad[1], [0., 0.])
    empty_loss, empty_grad = softmax_cross_entropy(np.zeros((0, 2)), np.zeros(0, dtype=int))
    ignored_loss, ignored_grad = softmax_cross_entropy(z, np.full(3, -100))
    assert empty_loss == ignored_loss == 0 and empty_grad.shape == (0, 2)
    assert np.array_equal(ignored_grad, np.zeros_like(z))


def verify_followup_validate_before_target_indexing(softmax_cross_entropy):
    for targets in (np.array([-1]), np.array([2]), np.array([0.5])):
        try:
            softmax_cross_entropy(np.zeros((1, 2)), targets)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid target accepted")


def verify_production_extreme_logits_and_row_shifts(softmax_cross_entropy):
    z = np.array([[1000., 1001.], [-1000., -999.]])
    y = np.array([1, 0])
    loss, grad = softmax_cross_entropy(z, y)
    shifted_loss, shifted_grad = softmax_cross_entropy(z + np.array([[9.], [-7.]]), y)
    assert np.isfinite(loss) and np.isclose(loss, shifted_loss)
    assert np.allclose(grad, shifted_grad) and np.allclose(grad.sum(axis=1), 0, atol=1e-14)


def verify_production_finite_difference_and_immutability(softmax_cross_entropy):
    z = np.array([[.2, -.4, 1.], [-.7, .3, .8]], dtype=np.float64)
    original = z.copy()
    y = np.array([2, 1])
    _, grad = softmax_cross_entropy(z, y)
    eps = 1e-6
    for i, j in ((0, 0), (0, 2), (1, 1)):
        plus, minus = z.copy(), z.copy()
        plus[i, j] += eps
        minus[i, j] -= eps
        numeric = (softmax_cross_entropy(plus, y)[0] - softmax_cross_entropy(minus, y)[0]) / (2 * eps)
        assert np.isclose(numeric, grad[i, j], atol=1e-8, rtol=1e-6)
    assert np.array_equal(z, original)


def verify_production_generated_shift_invariant(softmax_cross_entropy):
    @settings(max_examples=40, deadline=None, derandomize=True)
    @given(st.data())
    def property_check(data):
        n = data.draw(st.integers(1, 5))
        c = data.draw(st.integers(2, 6))
        values = data.draw(st.lists(st.floats(-20, 20, allow_nan=False), min_size=n*c, max_size=n*c))
        labels = data.draw(st.lists(st.integers(0, c-1), min_size=n, max_size=n))
        shifts = data.draw(st.lists(st.sampled_from([-1000., -3., 0., 5., 1000.]), min_size=n, max_size=n))
        z = np.asarray(values, dtype=np.float64).reshape(n, c)
        y = np.asarray(labels, dtype=np.int64)
        loss, grad = softmax_cross_entropy(z, y)
        shifted_loss, shifted_grad = softmax_cross_entropy(z + np.asarray(shifts)[:, None], y)
        assert np.isclose(loss, shifted_loss, atol=1e-12)
        assert np.allclose(grad, shifted_grad, atol=1e-12)
        assert np.allclose(grad.sum(axis=1), 0, atol=1e-12)
    property_check()


# %%
from torch_judge import evaluate
if not evaluate("0059_stable_cross_entropy_eval.py", softmax_cross_entropy):
    raise SystemExit(1)
