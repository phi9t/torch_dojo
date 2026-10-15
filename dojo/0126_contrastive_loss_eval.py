"""Reference and staged checks for the symmetric contrastive loss (NumPy)."""
# Drill answers: with X=Y=I(2), t=1 each row is -log(e/(e+1)) = 0.31326.
# logsumexp subtracts the row max, so exp never overflows even at t=1e-3.

# %%
import numpy as np


# %%
def contrastive_loss(X, Y, temperature, valid_mask, eps=1e-12):
    X, Y = np.asarray(X), np.asarray(Y)
    valid = np.asarray(valid_mask)
    if X.ndim != 2 or X.shape != Y.shape or valid.dtype != np.bool_ or valid.shape != (len(X),):
        raise ValueError("need X,Y [N,D] and bool valid_mask [N]")
    if not temperature > 0 or not np.isfinite(temperature) or not eps > 0:
        raise ValueError("temperature and eps must be positive and finite")
    X, Y = X[valid].astype(np.float64), Y[valid].astype(np.float64)  # drop BEFORE any math
    if not (np.isfinite(X).all() and np.isfinite(Y).all()):
        raise ValueError("valid embeddings must be finite")
    n = len(X)
    if n == 0:
        return 0.0
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), eps)
    Y = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), eps)
    logits = X @ Y.T / temperature  # [n,n]; row i = query i vs all candidates

    def cross_entropy(l):  # diagonal positives, stable logsumexp
        m = l.max(axis=1, keepdims=True)
        lse = m[:, 0] + np.log(np.exp(l - m).sum(axis=1))
        return float((lse - np.diag(l)).mean())

    return 0.5 * (cross_entropy(logits) + cross_entropy(logits.T))


# %% [verifiers]
def verify_baseline_identity_and_uniform_cases(contrastive_loss):
    I = np.eye(2)
    ok = np.ones(2, dtype=bool)
    assert abs(contrastive_loss(I, I, 1.0, ok) - 0.3132616875) < 1e-9
    same = np.tile([1.0, 2.0, 3.0], (5, 1))
    assert abs(contrastive_loss(same, same, 0.5, np.ones(5, dtype=bool)) - np.log(5)) < 1e-9


def verify_baseline_one_pair_and_empty_are_zero(contrastive_loss):
    X = np.arange(6.).reshape(3, 2)
    assert contrastive_loss(X, X + 1, 1.0, np.array([False, True, False])) == 0
    assert contrastive_loss(X, X, 1.0, np.zeros(3, dtype=bool)) == 0
    assert contrastive_loss(np.zeros((0, 2)), np.zeros((0, 2)), 1.0, np.zeros(0, dtype=bool)) == 0


def verify_followup_invalid_rows_leave_both_sets(contrastive_loss):
    rng = np.random.default_rng(0)
    X, Y = rng.normal(size=(5, 4)), rng.normal(size=(5, 4))
    mask = np.array([True, False, True, True, False])
    expected = contrastive_loss(X[mask], Y[mask], 0.3, np.ones(3, dtype=bool))
    assert abs(contrastive_loss(X, Y, 0.3, mask) - expected) < 1e-12
    X2, Y2 = X.copy(), Y.copy()
    X2[~mask], Y2[~mask] = np.nan, 1e30  # garbage in invalid rows must not leak
    assert abs(contrastive_loss(X2, Y2, 0.3, mask) - expected) < 1e-12


def verify_followup_zero_embeddings_and_permutation(contrastive_loss):
    Z = np.zeros((3, 2))
    value = contrastive_loss(Z, Z, 0.7, np.ones(3, dtype=bool))
    assert np.isfinite(value) and abs(value - np.log(3)) < 1e-9
    rng = np.random.default_rng(1)
    X, Y = rng.normal(size=(6, 3)), rng.normal(size=(6, 3))
    mask = np.array([True, True, False, True, True, True])
    p = rng.permutation(6)
    assert abs(contrastive_loss(X, Y, 0.5, mask) - contrastive_loss(X[p], Y[p], 0.5, mask[p])) < 1e-12


def verify_followup_symmetric_and_matches_naive_formula(contrastive_loss):
    rng = np.random.default_rng(2)
    X, Y = rng.normal(size=(4, 3)), rng.normal(size=(4, 3))
    ok = np.ones(4, dtype=bool)
    xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    yn = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    s = xn @ yn.T / 0.8
    naive = 0.5 * (-np.log(np.diag(np.exp(s) / np.exp(s).sum(1, keepdims=True))).mean()
                   - np.log(np.diag(np.exp(s.T) / np.exp(s.T).sum(1, keepdims=True))).mean())
    assert abs(contrastive_loss(X, Y, 0.8, ok) - naive) < 1e-9
    assert abs(contrastive_loss(X, Y, 0.8, ok) - contrastive_loss(Y, X, 0.8, ok)) < 1e-12


def verify_production_stable_at_tiny_temperature_and_scale_invariant(contrastive_loss):
    rng = np.random.default_rng(3)
    X, Y = rng.normal(size=(8, 5)), rng.normal(size=(8, 5))
    ok = np.ones(8, dtype=bool)
    tiny = contrastive_loss(X, Y, 1e-4, ok)
    assert np.isfinite(tiny) and tiny >= 0
    assert abs(contrastive_loss(1e6 * X, 1e-6 * Y, 0.2, ok) - contrastive_loss(X, Y, 0.2, ok)) < 1e-9
    assert contrastive_loss(np.eye(4), np.eye(4), 1e-4, np.ones(4, dtype=bool)) < 1e-6


def verify_production_rejects_invalid_contracts_and_does_not_mutate(contrastive_loss):
    X = np.ones((2, 3)); ok = np.ones(2, dtype=bool)
    for args in ((X, X, 0.0, ok), (X, X, -1.0, ok), (X, X, np.nan, ok), (X, np.ones((3, 3)), 1.0, ok),
                 (X, X, 1.0, np.ones(2)), (X, X, 1.0, np.ones(3, dtype=bool)),
                 (np.array([[np.inf, 1.], [1., 1.]]), np.ones((2, 2)), 1.0, ok)):
        try:
            contrastive_loss(*args)
        except ValueError:
            continue
        raise AssertionError("invalid input accepted")
    A, B = np.random.default_rng(4).normal(size=(3, 3)), np.random.default_rng(5).normal(size=(3, 3))
    a0, b0 = A.copy(), B.copy()
    contrastive_loss(A, B, 0.5, np.ones(3, dtype=bool))
    assert np.array_equal(A, a0) and np.array_equal(B, b0)


# %%
from torch_judge import evaluate
if not evaluate("0126_contrastive_loss_eval.py", contrastive_loss):
    raise SystemExit(1)
