"""Staged reference and verifiers for causal self-attention."""

# %%
import math
import torch

# %%
def causal_attention(Q, K, V, valid_keys=None):
    if Q.ndim != 3 or K.ndim != 3 or V.ndim != 3:
        raise ValueError("Q, K, V must be rank 3")
    b, s, d = Q.shape
    if (d == 0 or K.shape != (b, s, d) or V.shape[:2] != (b, s)
            or Q.dtype != K.dtype or Q.dtype != V.dtype or not Q.is_floating_point()
            or Q.device != K.device or Q.device != V.device):
        raise ValueError("incompatible attention inputs")
    if valid_keys is not None and (valid_keys.shape != (b, s) or valid_keys.dtype != torch.bool or valid_keys.device != Q.device):
        raise ValueError("valid_keys must be bool [B,S]")
    if s == 0:
        return V.clone()
    positions = torch.arange(s, device=Q.device)
    allowed = positions[None, :] <= positions[:, None]
    allowed = allowed[None, :, :]
    if valid_keys is not None:
        allowed = allowed & valid_keys[:, None, :]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d)
    active = allowed.any(dim=-1, keepdim=True)
    masked = scores.masked_fill(~allowed, -torch.inf)
    weights = torch.softmax(torch.where(active, masked, torch.zeros_like(masked)), dim=-1)
    weights = weights * allowed.to(weights.dtype)
    return torch.matmul(weights, V)

# %% [verifiers]
def verify_baseline_first_token(causal_attention):
    torch.manual_seed(9)
    q, k, v = (torch.randn(2, 5, 3, dtype=torch.float64) for _ in range(3))
    got = causal_attention(q, k, v)
    assert got.shape == v.shape and torch.allclose(got[:, 0], v[:, 0])


def verify_baseline_uniform_prefix(causal_attention):
    q = torch.zeros(1, 4, 2)
    k = torch.zeros_like(q)
    v = torch.arange(4.).reshape(1, 4, 1)
    got = causal_attention(q, k, v)[0, :, 0]
    assert torch.allclose(got, torch.tensor([0., .5, 1., 1.5]))


def verify_followup_future_independence(causal_attention):
    torch.manual_seed(2)
    q, k, v = (torch.randn(1, 6, 4) for _ in range(3))
    before = causal_attention(q, k, v)
    k[:, 4:] *= 1000
    v[:, 4:] *= 1000
    after = causal_attention(q, k, v)
    assert torch.allclose(before[:, :4], after[:, :4])


def verify_followup_padding_and_all_masked(causal_attention):
    q = torch.zeros(2, 3, 2)
    k = torch.zeros_like(q)
    v = torch.tensor([[[2.], [4.], [8.]], [[3.], [5.], [9.]]])
    valid = torch.tensor([[False, True, False], [False, False, False]])
    got = causal_attention(q, k, v, valid)
    assert torch.equal(got[0, :, 0], torch.tensor([0., 4., 4.]))
    assert torch.equal(got[1], torch.zeros_like(got[1]))


def verify_production_gradients_and_extremes(causal_attention):
    q = torch.tensor([[[1000., 0.], [0., 1000.]]], dtype=torch.float64, requires_grad=True)
    k = torch.tensor([[[1., 0.], [0., 1.]]], dtype=torch.float64, requires_grad=True)
    v = torch.tensor([[[2.], [7.]]], dtype=torch.float64, requires_grad=True)
    got = causal_attention(q, k, v)
    assert torch.isfinite(got).all() and torch.allclose(got[0, :, 0], torch.tensor([2., 7.], dtype=torch.float64))
    got.sum().backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in (q, k, v))


def verify_production_empty_and_validation(causal_attention):
    q = torch.empty(2, 0, 3)
    v = torch.empty(2, 0, 4)
    assert causal_attention(q, q, v).shape == (2, 0, 4)
    for args in ((torch.zeros(1, 2, 3), torch.zeros(1, 2, 2), torch.zeros(1, 2, 4)),
                 (torch.zeros(1, 2, 3), torch.zeros(1, 2, 3), torch.zeros(1, 2, 4), torch.ones(1, 2))):
        try:
            causal_attention(*args)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid causal contract accepted")

# %%
from torch_judge import evaluate
if not evaluate("0009_causal_attention_eval.py", causal_attention):
    raise SystemExit(1)
