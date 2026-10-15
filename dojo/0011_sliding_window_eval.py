"""Staged reference and verifiers for sliding-window attention."""

# %%
import math
import torch

# %%
def sliding_window_attention(Q, K, V, window_size, valid_keys=None):
    if Q.ndim != 3 or K.ndim != 3 or V.ndim != 3:
        raise ValueError("Q, K, V must be rank 3")
    b, s, d = Q.shape
    if (d == 0 or K.shape != (b, s, d) or V.shape[:2] != (b, s)
            or Q.dtype != K.dtype or Q.dtype != V.dtype or not Q.is_floating_point()
            or Q.device != K.device or Q.device != V.device):
        raise ValueError("incompatible attention inputs")
    if not isinstance(window_size, int) or isinstance(window_size, bool) or window_size < 0:
        raise ValueError("window_size must be a nonnegative integer")
    if valid_keys is not None and (valid_keys.shape != (b, s) or valid_keys.dtype != torch.bool or valid_keys.device != Q.device):
        raise ValueError("valid_keys must be bool [B,S]")
    if s == 0:
        return V.clone()
    outputs = []
    for i in range(s):
        lo, hi = max(0, i-window_size), min(s, i+window_size+1)
        scores = (Q[:, i:i+1] @ K[:, lo:hi].transpose(-2, -1)) / math.sqrt(d)
        if valid_keys is None:
            weights = torch.softmax(scores, dim=-1)
        else:
            allowed = valid_keys[:, None, lo:hi]
            active = allowed.any(dim=-1, keepdim=True)
            masked = scores.masked_fill(~allowed, -torch.inf)
            weights = torch.softmax(torch.where(active, masked, torch.zeros_like(masked)), dim=-1)
            weights = weights * allowed.to(weights.dtype)
        outputs.append(weights @ V[:, lo:hi])
    return torch.cat(outputs, dim=1)

# %% [verifiers]
def verify_baseline_zero_and_full_window(sliding_window_attention):
    q = torch.randn(2, 5, 3)
    k = torch.randn(2, 5, 3)
    v = torch.randn(2, 5, 2)
    assert torch.allclose(sliding_window_attention(q, k, v, 0), v)
    scores = q @ k.transpose(-2, -1) / math.sqrt(3)
    expected = torch.softmax(scores, -1) @ v
    assert torch.allclose(sliding_window_attention(q, k, v, 5), expected, atol=1e-6)


def verify_baseline_inclusive_boundaries(sliding_window_attention):
    q, k = torch.zeros(1, 5, 2), torch.zeros(1, 5, 2)
    v = torch.arange(5.).reshape(1, 5, 1)
    got = sliding_window_attention(q, k, v, 1)[0, :, 0]
    assert torch.allclose(got, torch.tensor([.5, 1., 2., 3., 3.5]))


def verify_followup_key_mask_and_empty_windows(sliding_window_attention):
    q = torch.zeros(1, 4, 2)
    k = torch.zeros_like(q)
    v = torch.tensor([[[2.], [4.], [8.], [16.]]])
    valid = torch.tensor([[False, False, True, False]])
    got = sliding_window_attention(q, k, v, 1, valid)
    assert torch.equal(got[0, :, 0], torch.tensor([0., 8., 8., 8.]))


def verify_followup_distant_keys_independent(sliding_window_attention):
    torch.manual_seed(11)
    q, k, v = (torch.randn(1, 9, 3) for _ in range(3))
    before = sliding_window_attention(q, k, v, 1)
    k[:, 5:] *= 100
    v[:, 5:] *= 100
    after = sliding_window_attention(q, k, v, 1)
    assert torch.allclose(before[:, :4], after[:, :4])


def verify_production_backward_and_empty(sliding_window_attention):
    q = torch.randn(2, 5, 3, dtype=torch.float64, requires_grad=True)
    k = torch.randn(2, 5, 3, dtype=torch.float64, requires_grad=True)
    v = torch.randn(2, 5, 2, dtype=torch.float64, requires_grad=True)
    out = sliding_window_attention(q, k, v, 2)
    out.square().sum().backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in (q, k, v))
    assert sliding_window_attention(q[:, :0], k[:, :0], v[:, :0], 1).shape == (2, 0, 2)


def verify_production_validation(sliding_window_attention):
    q = torch.zeros(1, 3, 2)
    v = torch.zeros(1, 3, 1)
    for args in ((q, q, v, -1), (q, q, v, 1.5),
                 (q, q[:, :2], v, 1),
                 (q, q, v, 1, torch.ones(1, 2, dtype=torch.bool))):
        try:
            sliding_window_attention(*args)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid window contract accepted")


def verify_production_large_sparse_window(sliding_window_attention):
    # A correct local implementation uses work proportional to the radius,
    # rather than a dense S-by-S attention score matrix.
    q = torch.zeros(1, 256, 2)
    k = torch.zeros_like(q)
    v = torch.arange(256.).reshape(1, 256, 1)
    got = sliding_window_attention(q, k, v, 1)
    assert got.shape == v.shape
    assert torch.allclose(got[0, 1:-1], v[0, 1:-1])

# %%
from torch_judge import evaluate
if not evaluate("0011_sliding_window_eval.py", sliding_window_attention):
    raise SystemExit(1)
