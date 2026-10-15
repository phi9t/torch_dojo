"""Staged reference and verifiers for scaled dot-product attention."""

# %%
import math
import torch

# %%
def scaled_dot_product_attention(Q, K, V, allowed=None):
    if Q.ndim != 3 or K.ndim != 3 or V.ndim != 3:
        raise ValueError("Q, K, V must be rank 3")
    b, tq, d = Q.shape
    if (d == 0 or K.shape[0] != b or V.shape[0] != b
            or K.shape[2] != d or K.shape[1] != V.shape[1] or K.shape[1] == 0):
        raise ValueError("incompatible attention shapes")
    if (Q.dtype != K.dtype or Q.dtype != V.dtype or not Q.is_floating_point()
            or Q.device != K.device or Q.device != V.device):
        raise ValueError("inputs need same floating dtype and device")
    if allowed is not None:
        if allowed.dtype != torch.bool or allowed.device != Q.device:
            raise ValueError("allowed must be bool on the input device")
        try:
            allowed = torch.broadcast_to(allowed, (b, tq, K.shape[1]))
        except RuntimeError as error:
            raise ValueError("mask is not broadcastable") from error
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d)
    if allowed is None:
        weights = torch.softmax(scores, dim=-1)
    else:
        active = allowed.any(dim=-1, keepdim=True)
        masked = scores.masked_fill(~allowed, -torch.inf)
        weights = torch.softmax(torch.where(active, masked, torch.zeros_like(masked)), dim=-1)
        weights = weights * allowed.to(weights.dtype)
    return torch.matmul(weights, V)

# %% [verifiers]
def verify_baseline_cross_attention(scaled_dot_product_attention):
    q = torch.zeros(2, 3, 4, dtype=torch.float64)
    k = torch.zeros(2, 5, 4, dtype=torch.float64)
    v = torch.arange(20., dtype=torch.float64).reshape(2, 5, 2)
    got = scaled_dot_product_attention(q, k, v)
    assert got.shape == (2, 3, 2)
    assert torch.allclose(got, v.mean(dim=1)[:, None, :].expand_as(got))


def verify_baseline_scale_and_key_axis(scaled_dot_product_attention):
    q = torch.tensor([[[2., 0.]]], dtype=torch.float64)
    k = torch.tensor([[[1., 0.], [0., 1.]]], dtype=torch.float64)
    v = torch.tensor([[[1.], [3.]]], dtype=torch.float64)
    p = torch.sigmoid(torch.tensor(2 / math.sqrt(2), dtype=torch.float64))
    assert torch.allclose(scaled_dot_product_attention(q, k, v), (p + (1-p)*3).reshape(1, 1, 1))


def verify_followup_mask_meaning_and_empty_rows(scaled_dot_product_attention):
    q = torch.zeros(1, 2, 2, dtype=torch.float64)
    k = torch.zeros(1, 3, 2, dtype=torch.float64)
    v = torch.tensor([[[2.], [5.], [11.]]], dtype=torch.float64)
    mask = torch.tensor([[[False, True, False], [False, False, False]]])
    got = scaled_dot_product_attention(q, k, v, mask)
    assert torch.equal(got, torch.tensor([[[5.], [0.]]], dtype=torch.float64))


def verify_followup_broadcast_mask(scaled_dot_product_attention):
    q = torch.zeros(2, 3, 2)
    k = torch.zeros(2, 3, 2)
    v = torch.arange(18., dtype=torch.float32).reshape(2, 3, 3)
    got = scaled_dot_product_attention(q, k, v, torch.tensor([True, False, False]))
    assert torch.equal(got, v[:, :1, :].expand_as(got))


def verify_production_batch_isolation(scaled_dot_product_attention):
    torch.manual_seed(6)
    q = torch.randn(4, 2, 3, dtype=torch.float64).transpose(0, 1)
    assert not q.is_contiguous()
    k = torch.randn(2, 5, 3, dtype=torch.float64)
    v = torch.randn(2, 5, 2, dtype=torch.float64)
    first = scaled_dot_product_attention(q, k, v)[0]
    k[1] *= 100
    v[1] *= 100
    assert torch.allclose(first, scaled_dot_product_attention(q, k, v)[0])


def verify_production_backward_and_extremes(scaled_dot_product_attention):
    q = torch.tensor([[[1000., 0.], [0., 1000.]]], dtype=torch.float64, requires_grad=True)
    k = torch.tensor([[[1., 0.], [0., 1.]]], dtype=torch.float64, requires_grad=True)
    v = torch.tensor([[[2.], [7.]]], dtype=torch.float64, requires_grad=True)
    got = scaled_dot_product_attention(q, k, v)
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0)
    assert torch.isfinite(got).all() and torch.allclose(got, expected)
    got.sum().backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in (q, k, v))
    q2 = torch.randn(1, 3, 4, dtype=torch.float64, requires_grad=True)
    k2 = torch.randn(1, 5, 4, dtype=torch.float64, requires_grad=True)
    v2 = torch.randn(1, 5, 2, dtype=torch.float64, requires_grad=True)
    actual = scaled_dot_product_attention(q2, k2, v2)
    reference = torch.nn.functional.scaled_dot_product_attention(q2, k2, v2, dropout_p=0)
    direction = torch.randn_like(actual)
    actual_grads = torch.autograd.grad((actual * direction).sum(), (q2, k2, v2), retain_graph=True)
    expected_grads = torch.autograd.grad((reference * direction).sum(), (q2, k2, v2))
    assert all(torch.allclose(a, b, atol=1e-10, rtol=1e-8) for a, b in zip(actual_grads, expected_grads))


def verify_production_shape_validation(scaled_dot_product_attention):
    q, k, v = torch.zeros(1, 2, 3), torch.zeros(1, 3, 3), torch.zeros(1, 3, 2)
    for args in ((q, k[:, :, :2], v), (q, k, v[:, :2]), (q, k, v, torch.ones(4, dtype=torch.bool))):
        try:
            scaled_dot_product_attention(*args)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid contract accepted")

# %%
from torch_judge import evaluate
if not evaluate("0005_attention_eval.py", scaled_dot_product_attention):
    raise SystemExit(1)
