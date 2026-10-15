"""Reference and staged checks for split-KV attention merged through logsumexp."""
# Probe answer: lse = log(e^1 + e^2) = 2.3133. Merging two halves with weights
# exp(lse_i - lse) gives the same result as one softmax over all the keys, because
# each half's output was normalised by exactly exp(lse_i).

# %%
import math
import torch


# %%
def split_kv_attention(Q, K, V, num_splits, allowed=None):
    if not isinstance(num_splits, int) or isinstance(num_splits, bool) or num_splits <= 0:
        raise ValueError("num_splits must be positive int")
    if Q.ndim != 3 or K.ndim != 3 or V.ndim != 3:
        raise ValueError("Q, K, V must be rank 3")
    b, tq, d = Q.shape
    tk = K.shape[1]
    if d == 0 or K.shape[0] != b or V.shape[0] != b or K.shape[2] != d or V.shape[1] != tk or tk == 0:
        raise ValueError("incompatible attention shapes")
    if (not Q.is_floating_point() or Q.dtype != K.dtype or Q.dtype != V.dtype
            or Q.device != K.device or Q.device != V.device):
        raise ValueError("inputs need same floating dtype and device")
    if allowed is not None:
        if allowed.dtype != torch.bool or allowed.device != Q.device:
            raise ValueError("allowed must be bool on the input device")
        try:
            allowed = torch.broadcast_to(allowed, (b, tq, tk))
        except RuntimeError as error:
            raise ValueError("mask is not broadcastable") from error
    scale = 1 / math.sqrt(d)
    edges = [i * tk // num_splits for i in range(num_splits + 1)]    # empty splits when num_splits > tk
    outs, lses = [], []
    for lo, hi in zip(edges, edges[1:]):
        if lo == hi:
            continue
        s = Q @ K[:, lo:hi].transpose(-2, -1) * scale
        if allowed is not None:
            s = s.masked_fill(~allowed[:, :, lo:hi], -torch.inf)
        m = s.max(-1, keepdim=True).values
        m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))   # fully masked split: avoid -inf - -inf
        e = torch.exp(s - m)
        mass = e.sum(-1, keepdim=True)
        live = mass > 0
        safe = torch.where(live, mass, torch.ones_like(mass))
        outs.append(e @ V[:, lo:hi] / safe)
        lses.append(torch.where(live, torch.log(safe) + m, torch.full_like(mass, -torch.inf)))
    top = torch.stack(lses).max(0).values
    top = torch.where(torch.isfinite(top), top, torch.zeros_like(top))
    weights = [torch.exp(l - top) for l in lses]                       # exp(-inf) = 0 for dead splits
    total = sum(weights)
    safe_total = torch.where(total > 0, total, torch.ones_like(total))
    return sum(w * o for w, o in zip(weights, outs)) / safe_total      # all-masked query: exactly 0


# %% [verifiers]
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves


def _dense(Q, K, V, allowed=None):
    s = Q @ K.transpose(-2, -1) / math.sqrt(Q.shape[-1])
    if allowed is None:
        return torch.softmax(s, -1) @ V
    allowed = torch.broadcast_to(allowed, s.shape)
    live = allowed.any(-1, keepdim=True)
    s = torch.where(live, s.masked_fill(~allowed, -torch.inf), torch.zeros_like(s))
    return (torch.softmax(s, -1) * allowed) @ V


def _reject(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError("invalid input accepted")


class _Peak(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.peak = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for t in tree_leaves(out):
            if isinstance(t, torch.Tensor):
                self.peak = max(self.peak, t.numel())
        return out


def _qkv(b, tq, tk, d, dv, seed, requires_grad=False):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(b, n, w, dtype=torch.float64, generator=g).requires_grad_(requires_grad)
            for n, w in ((tq, d), (tk, d), (tk, dv))]


def verify_baseline_matches_dense_for_any_split_count(split_kv_attention):
    q, k, v = _qkv(2, 5, 12, 4, 3, 1)
    for n in (1, 2, 3, 5, 12):
        assert torch.allclose(split_kv_attention(q, k, v, n), _dense(q, k, v), atol=1e-11), n


def verify_baseline_decode_step_with_uneven_splits(split_kv_attention):
    q, k, v = _qkv(3, 1, 37, 8, 8, 2)                  # one query token against a long cache
    assert split_kv_attention(q, k, v, 4).shape == (3, 1, 8)
    assert torch.allclose(split_kv_attention(q, k, v, 4), _dense(q, k, v), atol=1e-11)


def verify_followup_more_splits_than_keys(split_kv_attention):
    q, k, v = _qkv(1, 2, 3, 4, 2, 3)
    got = split_kv_attention(q, k, v, 8)
    assert torch.isfinite(got).all() and torch.allclose(got, _dense(q, k, v), atol=1e-11)


def verify_followup_masked_splits_and_empty_rows(split_kv_attention):
    q, k, v = _qkv(1, 3, 8, 4, 2, 4)
    mask = torch.zeros(1, 3, 8, dtype=torch.bool)
    mask[0, 0, 5] = True                               # one live key; splits of 2 keys leave others fully masked
    mask[0, 1, :] = True
    got = split_kv_attention(q, k, v, 4, mask)         # row 2 has no allowed key at all
    assert torch.isfinite(got).all()
    assert torch.allclose(got[0, 0], v[0, 5], atol=1e-12)
    assert torch.allclose(got, _dense(q, k, v, mask), atol=1e-11)
    assert torch.equal(got[0, 2], torch.zeros(2, dtype=torch.float64))


def verify_followup_broadcast_mask_matches_dense(split_kv_attention):
    q, k, v = _qkv(2, 4, 9, 3, 3, 5)
    mask = torch.arange(9) % 2 == 0
    assert torch.allclose(split_kv_attention(q, k, v, 3, mask), _dense(q, k, v, mask), atol=1e-11)


def verify_production_large_scores_stay_finite(split_kv_attention):
    q = torch.tensor([[[500., 0.]]], dtype=torch.float64)
    k = torch.tensor([[[1., 0.], [0., 1.], [-1., 0.], [1., 0.]]], dtype=torch.float64)
    v = torch.tensor([[[1.], [2.], [3.], [5.]]], dtype=torch.float64)
    got = split_kv_attention(q, k, v, 4)
    assert torch.isfinite(got).all() and torch.allclose(got, _dense(q, k, v), atol=1e-10)


def verify_production_gradients_match_dense_including_masked_rows(split_kv_attention):
    mask = torch.zeros(2, 3, 7, dtype=torch.bool)
    mask[:, :2, 1::2] = True                           # row 2 fully masked
    mine = _qkv(2, 3, 7, 4, 3, 6, requires_grad=True)
    theirs = [t.detach().clone().requires_grad_() for t in mine]
    w = torch.randn(2, 3, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(8))
    (split_kv_attention(*mine, 3, mask) * w).sum().backward()
    (_dense(*theirs, mask) * w).sum().backward()
    for a, b in zip(mine, theirs):
        assert a.grad is not None and torch.isfinite(a.grad).all()
        assert torch.allclose(a.grad, b.grad, atol=1e-10)


def verify_production_each_split_is_smaller_than_the_full_score_matrix(split_kv_attention):
    tq, tk = 64, 1024
    q, k, v = _qkv(1, tq, tk, 4, 4, 9)
    with _Peak() as probe:
        got = split_kv_attention(q, k, v, 16)
    assert probe.peak < tq * tk // 2, f"an operation produced {probe.peak} elements; full scores are {tq * tk}"
    assert torch.allclose(got, _dense(q, k, v), atol=1e-10)


def verify_production_validation_and_noncontiguous(split_kv_attention):
    q, k, v = _qkv(1, 2, 4, 3, 2, 10)
    for bad in (0, -1, 1.5, True):
        _reject(lambda: split_kv_attention(q, k, v, bad))
    _reject(lambda: split_kv_attention(q, k[:, :0], v[:, :0], 2))
    _reject(lambda: split_kv_attention(q, k, v[:, :2], 2))
    _reject(lambda: split_kv_attention(q.float(), k, v, 2))
    _reject(lambda: split_kv_attention(q, k, v, 2, torch.ones(2, 4)))
    _reject(lambda: split_kv_attention(q, k, v, 2, torch.ones(5, dtype=torch.bool)))
    q2, k2, v2 = _qkv(2, 4, 6, 3, 2, 11)
    assert torch.allclose(split_kv_attention(q2, k2, v2, 2), _dense(q2, k2, v2), atol=1e-11)
    qs = torch.randn(4, 2, 3, dtype=torch.float64).transpose(0, 1)
    assert not qs.is_contiguous()
    assert torch.allclose(split_kv_attention(qs, k2, v2, 3), _dense(qs, k2, v2), atol=1e-11)


# %%
from torch_judge import evaluate
if not evaluate("0123_split_kv_attention_eval.py", split_kv_attention):
    raise SystemExit(1)
