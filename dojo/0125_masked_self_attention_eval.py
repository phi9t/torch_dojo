"""Reference and staged checks for masked multi-head self-attention (NumPy)."""
# Drill answers: scores are [B,H,T,T]; with Wq=Wk=0 every score is 0, so each
# valid query averages the valid values uniformly. A row with no allowed key
# has max -inf, and -inf - (-inf) = NaN, so make such rows safe BEFORE softmax.
#
# Mistakes seen on this problem (each one was made while solving it):
#
# 1. Skipping the head split. `(X @ Wk)` gives [B,T,D]; scoring it directly
#    yields [B,T,T], one head's worth. Reshape to [B,T,H,dh] and transpose to
#    [B,H,T,dh] first. Scale by sqrt(dh), not sqrt(D).
# 2. Misnamed variable in the causal branch (`score` vs `full_score`): a
#    NameError that only appears when causal=True, so the unmasked path looks
#    fine. Build the mask once, then `if causal: allowed &= tril`, to avoid
#    two near-identical branches.
# 3. Masking only half of the softmax. Setting masked scores to -inf and
#    zeroing the exponentials is not enough: a row with NO allowed key has
#    sum 0, so `weights /= weights.sum(-1)` is 0/0 = NaN. Two fixes together:
#    replace such rows' scores with 0 before the max/exp, and divide by
#    `np.where(has_key, sum, 1.0)`. Invalid QUERY rows and the first row of a
#    causal sequence whose first token is invalid both hit this.
# 4. `assert` instead of `ValueError`. The contract asks for ValueError, and
#    asserts vanish under `python -O`. Validate ndim, num_heads | D, weight
#    shapes [D,D] and a bool [B,T] valid_mask.
# 5. Multiplying by the mask instead of selecting. `NaN * 0` is NaN; use
#    np.where so garbage in masked positions cannot leak into the result.
# 6. In-place ops on inputs (`weights /= ...` on an array you also return is
#    fine, but `X /= ...` or writing into valid_mask mutates the caller's
#    data). Production stage snapshots X, the weights and the mask.
# 7. Hard-coding Tq == Tk. Every reshape and the [T,T] mask use one length;
#    cross-attention needs separate Tq and Tk (extension, not graded).

# %%
import numpy as np


# %%
def self_attention(X, Wq, Wk, Wv, Wo, num_heads, valid_mask, causal=False):
    X, Wq, Wk, Wv, Wo = (np.asarray(a, dtype=np.float64) for a in (X, Wq, Wk, Wv, Wo))
    valid = np.asarray(valid_mask)
    if X.ndim != 3:
        raise ValueError("X must be [B,T,D]")
    B, T, D = X.shape
    if not isinstance(num_heads, (int, np.integer)) or num_heads < 1 or D % num_heads:
        raise ValueError("num_heads must be a positive divisor of D")
    if any(w.shape != (D, D) for w in (Wq, Wk, Wv, Wo)):
        raise ValueError("weights must be [D,D]")
    if valid.dtype != np.bool_ or valid.shape != (B, T):
        raise ValueError("valid_mask must be bool [B,T]")
    H, dh = int(num_heads), D // int(num_heads)

    def split(M):  # [B,T,D] -> [B,H,T,dh]
        return (X @ M).reshape(B, T, H, dh).transpose(0, 2, 1, 3)

    q, k, v = split(Wq), split(Wk), split(Wv)
    scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(dh)  # [B,H,T,T]
    allowed = valid[:, None, :, None] & valid[:, None, None, :]  # query & key valid
    if causal:
        allowed = allowed & np.tril(np.ones((T, T), dtype=bool))
    has_key = allowed.any(axis=-1, keepdims=True)  # [B,1,T,1]
    safe = np.where(allowed, scores, -np.inf)
    safe = np.where(has_key, safe, 0.0)  # empty rows never see an all -inf softmax
    e = np.exp(safe - safe.max(axis=-1, keepdims=True)) * allowed
    denom = np.where(has_key, e.sum(axis=-1, keepdims=True), 1.0)
    weights = np.broadcast_to(e / denom, (B, H, T, T)).copy()
    context = (weights @ v).transpose(0, 2, 1, 3).reshape(B, T, D)
    return context @ Wo, weights


# %% [verifiers]
I2 = np.eye(2)
Z2 = np.zeros((2, 2))
X2 = np.array([[[1., 0.], [0., 2.]]])
ALL2 = np.ones((1, 2), dtype=bool)


def verify_baseline_one_valid_token_has_weight_one(self_attention):
    X = np.random.default_rng(0).normal(size=(1, 3, 4))
    W = [np.random.default_rng(i).normal(size=(4, 4)) for i in range(1, 5)]
    out, w = self_attention(X, *W, 2, np.array([[False, True, False]]), False)
    assert out.shape == (1, 3, 4) and w.shape == (1, 2, 3, 3)
    assert np.allclose(w[0, :, 1, 1], 1.0) and np.allclose(w[0, :, 1].sum(-1), 1.0)
    assert np.all(w[0, :, 0] == 0) and np.all(w[0, :, 2] == 0)


def verify_baseline_uniform_attention_examples(self_attention):
    out, w = self_attention(X2, Z2, Z2, I2, I2, 1, ALL2, False)
    assert np.allclose(out, [[[0.5, 1.0], [0.5, 1.0]]]) and np.allclose(w, 0.5)
    out, w = self_attention(X2, Z2, Z2, I2, I2, 1, ALL2, True)
    assert np.allclose(out, [[[1.0, 0.0], [0.5, 1.0]]])
    assert np.allclose(w[0, 0], [[1.0, 0.0], [0.5, 0.5]])


def verify_followup_all_invalid_batch_element_is_zero(self_attention):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(2, 3, 4))
    W = [rng.normal(size=(4, 4)) for _ in range(4)]
    mask = np.array([[True, True, True], [False, False, False]])
    for causal in (False, True):
        out, w = self_attention(X, *W, 2, mask, causal)
        assert np.isfinite(out).all() and np.isfinite(w).all()
        assert np.all(out[1] == 0) and np.all(w[1] == 0)
        assert np.allclose(w[0].sum(-1), 1.0)


def verify_followup_invalid_tokens_are_not_attended(self_attention):
    rng = np.random.default_rng(2)
    X = rng.normal(size=(1, 4, 4))
    W = [rng.normal(size=(4, 4)) for _ in range(4)]
    mask = np.array([[True, False, True, True]])
    out, w = self_attention(X, *W, 2, mask, False)
    changed = X.copy(); changed[0, 1] += 100
    out2, _ = self_attention(changed, *W, 2, mask, False)
    assert np.all(w[..., 1] == 0) and np.all(w[:, :, 1, :] == 0) and np.all(out[0, 1] == 0)
    assert np.allclose(out[0, [0, 2, 3]], out2[0, [0, 2, 3]])


def verify_followup_future_token_does_not_change_causal_past(self_attention):
    rng = np.random.default_rng(3)
    X = rng.normal(size=(2, 5, 6))
    W = [rng.normal(size=(6, 6)) for _ in range(4)]
    mask = np.ones((2, 5), dtype=bool)
    out, w = self_attention(X, *W, 3, mask, True)
    changed = X.copy(); changed[:, 3:] += 50
    out2, _ = self_attention(changed, *W, 3, mask, True)
    assert np.allclose(out[:, :3], out2[:, :3])
    assert np.all(np.triu(w[0, 0], 1) == 0)


def verify_followup_matches_independent_loop_reference(self_attention):
    rng = np.random.default_rng(4)
    B, T, D, H = 2, 4, 6, 2
    X = rng.normal(size=(B, T, D))
    Wq, Wk, Wv, Wo = (rng.normal(size=(D, D)) for _ in range(4))
    mask = np.array([[True, True, False, True], [True, True, True, True]])
    for causal in (False, True):
        out, w = self_attention(X, Wq, Wk, Wv, Wo, H, mask, causal)
        dh = D // H
        for b in range(B):
            heads = np.zeros((T, D))
            for h in range(H):
                s = slice(h * dh, (h + 1) * dh)
                q, k, v = (X[b] @ M[:, s] for M in (Wq, Wk, Wv))
                for i in range(T):
                    keys = [j for j in range(T) if mask[b, j] and mask[b, i] and (not causal or j <= i)]
                    if not keys:
                        assert np.all(w[b, h, i] == 0)
                        continue
                    logits = np.array([q[i] @ k[j] for j in keys]) / np.sqrt(dh)
                    p = np.exp(logits - logits.max()); p /= p.sum()
                    heads[i, s] = sum(pj * v[j] for pj, j in zip(p, keys))
                    assert np.allclose(w[b, h, i, keys], p)
            assert np.allclose(out[b], heads @ Wo)


def verify_production_stable_finite_and_pure(self_attention):
    rng = np.random.default_rng(5)
    X = 300 * rng.normal(size=(2, 4, 4))
    W = [rng.normal(size=(4, 4)) for _ in range(4)]
    mask = np.array([[True, False, False, False], [False, False, False, False]])
    snapshot = [a.copy() for a in (X, *W, mask)]
    for causal in (False, True):
        out, w = self_attention(X, *W, 2, mask, causal)
        assert out.dtype == np.float64 and w.dtype == np.float64
        assert np.isfinite(out).all() and np.isfinite(w).all()
        assert np.all(out[1] == 0) and np.allclose(w[0, :, 0, 0], 1.0)
    for before, after in zip(snapshot, (X, *W, mask)):
        assert np.array_equal(before, after)


def verify_production_rejects_invalid_contracts(self_attention):
    X, W = np.zeros((1, 2, 4)), [np.eye(4)] * 4
    good = np.ones((1, 2), dtype=bool)
    for args in ((X, *W, 3, good, False), (X, *W, 0, good, False),
                 (X, *W, 2, np.ones((1, 2)), False), (X, *W, 2, np.ones((1, 3), dtype=bool), False),
                 (X, np.eye(3), *W[1:], 2, good, False), (np.zeros((2, 4)), *W, 2, good, False)):
        try:
            self_attention(*args)
        except ValueError:
            continue
        raise AssertionError("invalid input accepted")


# %%
from torch_judge import evaluate
if not evaluate("0125_masked_self_attention_eval.py", self_attention):
    raise SystemExit(1)
