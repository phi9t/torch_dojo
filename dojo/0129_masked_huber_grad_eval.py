"""Reference and staged checks for masked Huber loss with a manual gradient."""
# Drill answers: d/dr of Huber is clip(r, -delta, delta). Multiplying by the
# mask is NOT enough: NaN * 0 = NaN, so SELECT with np.where instead.

# %%
import numpy as np


# %%
def huber_loss_and_grad(prediction, target, valid_mask, delta):
    prediction, target = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
    valid = np.asarray(valid_mask)
    if prediction.ndim != 2 or prediction.shape != target.shape:
        raise ValueError("prediction and target must be [N,D]")
    if valid.dtype != np.bool_ or valid.shape != (len(prediction),):
        raise ValueError("valid_mask must be bool [N]")
    if not delta > 0 or not np.isfinite(delta):
        raise ValueError("delta must be positive and finite")
    n_valid = int(valid.sum())
    grad = np.zeros_like(prediction)
    if n_valid == 0 or prediction.shape[1] == 0:
        return 0.0, grad
    count = n_valid * prediction.shape[1]
    residual = np.where(valid[:, None], prediction - target, 0.0)  # select, don't multiply
    quad = np.abs(residual) <= delta
    per_element = np.where(quad, 0.5 * residual ** 2, delta * (np.abs(residual) - 0.5 * delta))
    grad = np.clip(residual, -delta, delta) / count
    return float(per_element.sum() / count), grad


# %% [verifiers]
def verify_baseline_worked_example(huber_loss_and_grad):
    loss, grad = huber_loss_and_grad(np.array([[-2., 0.5]]), np.zeros((1, 2)), np.array([True]), 1.0)
    assert abs(loss - 0.8125) < 1e-12 and grad.shape == (1, 2)
    assert np.allclose(grad, [[-0.5, 0.25]])


def verify_baseline_exact_prediction_is_zero(huber_loss_and_grad):
    x = np.arange(6.).reshape(3, 2)
    loss, grad = huber_loss_and_grad(x, x.copy(), np.ones(3, dtype=bool), 2.0)
    assert loss == 0 and np.all(grad == 0)


def verify_followup_both_sides_of_the_threshold(huber_loss_and_grad):
    for r, expected_loss, expected_grad in ((0.99, 0.5 * 0.99 ** 2, 0.99), (1.0, 0.5, 1.0),
                                            (1.01, 1.01 - 0.5, 1.0), (-3.0, 2.5, -1.0)):
        loss, grad = huber_loss_and_grad(np.array([[r]]), np.zeros((1, 1)), np.array([True]), 1.0)
        assert abs(loss - expected_loss) < 1e-12 and abs(grad[0, 0] - expected_grad) < 1e-12, r


def verify_followup_masked_rows_and_means(huber_loss_and_grad):
    pred = np.array([[1., 2.], [100., -100.], [0., 4.]])
    target = np.zeros((3, 2))
    mask = np.array([True, False, True])
    loss, grad = huber_loss_and_grad(pred, target, mask, 1.5)
    kept = huber_loss_and_grad(pred[mask], target[mask], np.ones(2, dtype=bool), 1.5)
    assert abs(loss - kept[0]) < 1e-12 and np.allclose(grad[mask], kept[1])
    assert np.all(grad[1] == 0)
    assert huber_loss_and_grad(pred, target, np.zeros(3, dtype=bool), 1.0)[0] == 0
    empty_loss, empty_grad = huber_loss_and_grad(np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0, dtype=bool), 1.0)
    assert empty_loss == 0 and empty_grad.shape == (0, 2)


def verify_production_matches_float64_finite_differences(huber_loss_and_grad):
    rng = np.random.default_rng(0)
    pred, target = 2 * rng.normal(size=(4, 3)), rng.normal(size=(4, 3))
    mask = np.array([True, True, False, True])
    delta = 0.9
    _, grad = huber_loss_and_grad(pred, target, mask, delta)
    step = 1e-6
    for index in np.ndindex(*pred.shape):
        up, down = pred.copy(), pred.copy()
        up[index] += step; down[index] -= step
        numeric = (huber_loss_and_grad(up, target, mask, delta)[0] - huber_loss_and_grad(down, target, mask, delta)[0]) / (2 * step)
        assert abs(numeric - grad[index]) < 1e-6, index


def verify_production_invalid_rows_may_contain_garbage(huber_loss_and_grad):
    pred = np.array([[1., 2.], [np.nan, np.inf], [3., 0.]])
    target = np.array([[0., 0.], [np.nan, 1e300], [0., 0.]])
    mask = np.array([True, False, True])
    loss, grad = huber_loss_and_grad(pred, target, mask, 1.0)
    assert np.isfinite(loss) and np.isfinite(grad).all() and np.all(grad[1] == 0)


def verify_production_rejects_invalid_contracts_and_is_pure(huber_loss_and_grad):
    p, t, m = np.ones((2, 2)), np.zeros((2, 2)), np.ones(2, dtype=bool)
    for args in ((p, t, m, 0.0), (p, t, m, -1.0), (p, np.zeros((2, 3)), m, 1.0), (p, t, np.ones(2), 1.0),
                 (p, t, np.ones(3, dtype=bool), 1.0), (p, t, m, np.nan)):
        try:
            huber_loss_and_grad(*args)
        except ValueError:
            continue
        raise AssertionError("invalid input accepted")
    p0, t0 = p.copy(), t.copy()
    huber_loss_and_grad(p, t, m, 1.0)
    assert np.array_equal(p, p0) and np.array_equal(t, t0)


# %%
from torch_judge import evaluate
if not evaluate("0129_masked_huber_grad_eval.py", huber_loss_and_grad):
    raise SystemExit(1)
