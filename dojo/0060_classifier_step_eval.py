"""Three-stage reference for one stable full-batch classifier update."""

# %%
import numpy as np

# %%
# Drill 1: Each row of `probs - onehot` sums to zero, because a softmax row and
# a one-hot row both sum to 1. So `dZ.sum(axis=1)` prints `[0. 0. 0.]` for any
# batch; it carries no information. Summing over examples (`axis=0`) leaves one
# entry per class, $[1/6, -1/6]$, which has the shape $(C,)$ of `b`.
#
# Drill 2: The function receives the caller's array, not a copy, and `-=` on a
# NumPy array writes into that buffer. The caller's `W` prints `[-0.5 -0.5]`
# and `new_W is W` prints `True`: the "new" weights and the input are one
# object. `W - 0.5 * dW` would allocate a fresh array and leave the input alone.
#
# Drill 3: Averaging the two means prints `2.0`, because it gives the
# one-example microbatch the same weight as the three-example one. The
# full-batch loss is a mean over examples, `1.0`; recover it by weighting each
# microbatch by its valid count: $(1 \cdot 4 + 3 \cdot 0)/4 = 1$.

# %%
def classifier_step(X: np.ndarray, targets: np.ndarray, W: np.ndarray,
                    b: np.ndarray, learning_rate: float,
                    ignore_index: int = -100) -> tuple[np.ndarray, np.ndarray, float]:
    X, targets, W, b = map(np.asarray, (X, targets, W, b))
    if (X.ndim != 2 or W.ndim != 2 or W.shape[1] < 2 or X.shape[1] != W.shape[0]
            or b.shape != (W.shape[1],) or targets.shape != (len(X),)):
        raise ValueError("incompatible classifier shapes")
    if any(value.dtype != np.float64 or not np.isfinite(value).all() for value in (X, W, b)):
        raise ValueError("X, W, and b must be finite float64 arrays")
    if targets.dtype.kind not in "iu" or not isinstance(ignore_index, (int, np.integer)):
        raise ValueError("integer targets and ignore_index required")
    if not np.isfinite(learning_rate) or learning_rate < 0:
        raise ValueError("learning_rate must be finite and nonnegative")
    valid = targets != ignore_index
    if np.any((targets[valid] < 0) | (targets[valid] >= W.shape[1])):
        raise ValueError("invalid class target")
    count = int(valid.sum())
    if count == 0:
        return W.copy(), b.copy(), 0.0
    logits = X[valid] @ W + b
    q = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(q)
    mass = exponentials.sum(axis=1, keepdims=True)
    probabilities = exponentials / mass
    row = np.arange(count)
    labels = targets[valid]
    loss = float(np.mean(np.log(mass[:, 0]) - q[row, labels]))
    probabilities[row, labels] -= 1
    probabilities /= count
    dW = X[valid].T @ probabilities
    db = probabilities.sum(axis=0)
    return W - learning_rate * dW, b - learning_rate * db, loss


# %% [verifiers]
def verify_baseline_one_step_and_preupdate_loss(classifier_step):
    X = np.eye(2, dtype=np.float64)
    y = np.array([0, 1])
    W, b = np.zeros((2, 2)), np.zeros(2)
    new_W, new_b, loss = classifier_step(X, y, W, b, 1.)
    assert np.isclose(loss, np.log(2))
    assert new_W[0, 0] > new_W[0, 1] and new_W[1, 1] > new_W[1, 0]
    assert np.array_equal(new_b, b)


def verify_baseline_loss_decreases(classifier_step):
    X = np.array([[2., 0.], [0., 2.], [-2., 0.], [0., -2.]])
    y = np.array([0, 1, 1, 0])
    W, b = np.zeros((2, 2)), np.zeros(2)
    losses = []
    for _ in range(20):
        W, b, loss = classifier_step(X, y, W, b, .2)
        losses.append(loss)
    assert losses[-1] < losses[0] * .5


def verify_followup_ignore_and_all_ignored(classifier_step):
    X = np.array([[1., 0.], [10., 10.], [0., 1.]])
    y = np.array([0, -100, 1])
    W, b = np.zeros((2, 2)), np.zeros(2)
    a, c, loss = classifier_step(X, y, W, b, .5)
    expected_a, expected_c, expected_loss = classifier_step(X[[0, 2]], np.array([0, 1]), W, b, .5)
    assert np.allclose(a, expected_a) and np.allclose(c, expected_c) and np.isclose(loss, expected_loss)
    empty = classifier_step(np.zeros((0, 2)), np.zeros(0, dtype=int), W, b, .5)
    ignored = classifier_step(X, np.full(3, -100), W, b, .5)
    for new_W, new_b, zero_loss in (empty, ignored):
        assert zero_loss == 0 and np.array_equal(new_W, W) and np.array_equal(new_b, b)
        assert new_W is not W and new_b is not b


def verify_followup_zero_rate_returns_copies(classifier_step):
    W, b = np.ones((1, 2)), np.ones(2)
    new_W, new_b, _ = classifier_step(np.ones((1, 1)), np.array([0]), W, b, 0.)
    assert np.array_equal(new_W, W) and np.array_equal(new_b, b)
    assert new_W is not W and new_b is not b


def verify_production_weight_unequal_microbatches(classifier_step):
    X = np.array([[1., 0.], [0., 1.], [2., 1.], [5., 5.]])
    y = np.array([0, 1, 0, -100])
    W = np.array([[.2, -.1], [-.3, .4]])
    b = np.array([.1, -.2])
    rate = .3
    full_W, full_b, _ = classifier_step(X, y, W, b, rate)
    first_W, first_b, _ = classifier_step(X[:1], y[:1], W, b, rate / 3)
    second_W, second_b, _ = classifier_step(X[1:], y[1:], W, b, rate * 2 / 3)
    assert np.allclose(full_W, W + (first_W - W) + (second_W - W), atol=1e-12)
    assert np.allclose(full_b, b + (first_b - b) + (second_b - b), atol=1e-12)


def verify_production_validation_and_no_mutation(classifier_step):
    X = np.array([[1., 2.]])
    W = np.zeros((2, 2)); b = np.zeros(2); y = np.array([0])
    old = [a.copy() for a in (X, W, b)]
    classifier_step(X, y, W, b, .1)
    assert all(np.array_equal(a, before) for a, before in zip((X, W, b), old))
    invalid = [(X, np.array([2]), W, b, .1),
               (X, y, W, b, -1.),
               (X, y, W, b, float("nan")),
               (X, y, W, np.array([0., np.inf]), .1),
               (X.astype(np.float32), y, W, b, .1)]
    for args in invalid:
        try:
            classifier_step(*args)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid classifier input accepted")


# %%
from torch_judge import evaluate
if not evaluate("0060_classifier_step_eval.py", classifier_step):
    raise SystemExit(1)
