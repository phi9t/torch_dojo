"""Reference and staged checks for packed variable-length attention."""
# Probe answer: cu_seqlens = [0, 3, 5, 9] gives lengths [3, 2, 4]. Token 4 (the last
# of sequence 1) may attend to tokens 3 and 4 only, so its row has 2 allowed keys;
# the block-diagonal mask has 3*3 + 2*2 + 4*4 = 29 True entries out of 81.

# %%
import math
import torch


# %%
def packed_attention(q, k, v, cu_seqlens, causal=False):
    if q.ndim != 2 or k.ndim != 2 or v.ndim != 2:
        raise ValueError("q, k, v must be rank 2 [N, width]")
    n, d = q.shape
    if k.shape != (n, d) or v.shape[0] != n or d == 0:
        raise ValueError("q, k, v must be packed with the same total length N")
    if (not q.is_floating_point() or q.dtype != k.dtype or q.dtype != v.dtype
            or q.device != k.device or q.device != v.device):
        raise ValueError("inputs need same floating dtype and device")
    if not isinstance(causal, bool):
        raise ValueError("causal must be bool")
    if (cu_seqlens.ndim != 1 or cu_seqlens.dtype not in (torch.int32, torch.int64)
            or cu_seqlens.device != q.device or cu_seqlens.numel() < 1):
        raise ValueError("cu_seqlens must be a 1-D int32/int64 tensor on the input device")
    bounds = cu_seqlens.tolist()
    if bounds[0] != 0 or bounds[-1] != n or any(b < a for a, b in zip(bounds, bounds[1:])):
        raise ValueError("cu_seqlens must start at 0, end at N and never decrease")
    scale = 1 / math.sqrt(d)
    chunks = []
    for lo, hi in zip(bounds, bounds[1:]):
        if lo == hi:
            continue
        s = q[lo:hi] @ k[lo:hi].transpose(0, 1) * scale
        if causal:
            s = s.masked_fill(torch.ones_like(s, dtype=torch.bool).triu(1), -torch.inf)
        chunks.append(torch.softmax(s, dim=-1) @ v[lo:hi])
    return torch.cat(chunks) if chunks else v.new_empty((0, v.shape[-1]))


# %% [verifiers]
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves


def _dense_one(q, k, v, causal=False):
    s = q @ k.transpose(0, 1) / math.sqrt(q.shape[-1])
    if causal:
        s = s.masked_fill(torch.ones_like(s, dtype=torch.bool).triu(1), -torch.inf)
    return torch.softmax(s, -1) @ v


def _per_sequence(q, k, v, lengths, causal=False):
    out, start = [], 0
    for n in lengths:
        if n:
            out.append(_dense_one(q[start:start + n], k[start:start + n], v[start:start + n], causal))
        start += n
    return torch.cat(out) if out else v.new_empty((0, v.shape[-1]))


def _cu(lengths, dtype=torch.int32):
    return torch.tensor([0] + list(torch.tensor(lengths).cumsum(0).tolist()), dtype=dtype)


def _packed(lengths, seed, d=4, dv=3, requires_grad=False):
    g = torch.Generator().manual_seed(seed)
    n = sum(lengths)
    return [torch.randn(n, w, dtype=torch.float64, generator=g).requires_grad_(requires_grad)
            for w in (d, d, dv)]


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


def verify_baseline_matches_per_sequence_attention(packed_attention):
    lengths = [3, 2, 4]
    q, k, v = _packed(lengths, 1)
    assert torch.allclose(packed_attention(q, k, v, _cu(lengths)), _per_sequence(q, k, v, lengths), atol=1e-11)


def verify_baseline_no_leakage_across_sequences(packed_attention):
    lengths = [3, 4]
    q, k, v = _packed(lengths, 2)
    first = packed_attention(q, k, v, _cu(lengths))[:3]
    k2, v2 = k.clone(), v.clone()
    k2[3:] *= 100
    v2[3:] += 50
    assert torch.allclose(first, packed_attention(q, k2, v2, _cu(lengths))[:3])


def verify_followup_causal_restarts_in_each_sequence(packed_attention):
    lengths = [3, 2]
    q, k, v = _packed(lengths, 3)
    got = packed_attention(q, k, v, _cu(lengths), causal=True)
    assert torch.allclose(got, _per_sequence(q, k, v, lengths, causal=True), atol=1e-11)
    assert torch.allclose(got[3], v[3], atol=1e-12)        # first token of sequence 2 sees only itself
    assert torch.allclose(got[0], v[0], atol=1e-12)


def verify_followup_empty_and_single_token_sequences(packed_attention):
    lengths = [0, 1, 0, 3, 0]
    q, k, v = _packed(lengths, 4)
    cu = _cu(lengths, torch.int64)
    got = packed_attention(q, k, v, cu)
    assert got.shape == (4, 3) and torch.allclose(got[0], v[0], atol=1e-12)
    assert torch.allclose(got, _per_sequence(q, k, v, lengths), atol=1e-11)
    none = packed_attention(q[:0], k[:0], v[:0], torch.zeros(3, dtype=torch.int32))
    assert none.shape == (0, 3)


def verify_production_validates_cu_seqlens_and_inputs(packed_attention):
    q, k, v = _packed([2, 3], 5)
    ok = _cu([2, 3])
    _reject(lambda: packed_attention(q, k, v, torch.tensor([1, 2, 5], dtype=torch.int32)))   # must start at 0
    _reject(lambda: packed_attention(q, k, v, torch.tensor([0, 2, 4], dtype=torch.int32)))   # must end at N
    _reject(lambda: packed_attention(q, k, v, torch.tensor([0, 4, 2, 5], dtype=torch.int32)))  # decreasing
    _reject(lambda: packed_attention(q, k, v, ok.float()))
    _reject(lambda: packed_attention(q, k, v, ok.reshape(1, 3)))
    _reject(lambda: packed_attention(q, k[:4], v, ok))
    _reject(lambda: packed_attention(q, k, v[:4], ok))
    _reject(lambda: packed_attention(q, k, v.float(), ok))
    _reject(lambda: packed_attention(q[None], k, v, ok))
    _reject(lambda: packed_attention(q, k, v, ok, causal=1))


def verify_production_gradients_match_per_sequence_autograd(packed_attention):
    lengths = [4, 1, 3]
    for causal in (False, True):
        mine = _packed(lengths, 6, requires_grad=True)
        theirs = [t.detach().clone().requires_grad_() for t in mine]
        w = torch.randn(8, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(7))
        (packed_attention(*mine, _cu(lengths), causal=causal) * w).sum().backward()
        (_per_sequence(*theirs, lengths, causal) * w).sum().backward()
        for a, b in zip(mine, theirs):
            assert a.grad is not None and torch.allclose(a.grad, b.grad, atol=1e-10)


def verify_production_workspace_is_per_sequence_not_total_squared(packed_attention):
    lengths = [16] * 63 + [1000]                           # padding to the longest would cost 64 * 1000^2
    q, k, v = _packed(lengths, 8)
    n = sum(lengths)
    with _Peak() as probe:
        got = packed_attention(q, k, v, _cu(lengths))
    assert probe.peak < n * n // 2, f"an operation produced {probe.peak} elements; N^2 is {n * n}"
    assert torch.allclose(got[:16], _dense_one(q[:16], k[:16], v[:16]), atol=1e-10)


def verify_production_noncontiguous_inputs(packed_attention):
    lengths = [3, 3]
    q, k, v = _packed(lengths, 9)
    qt = q.transpose(0, 1).contiguous().transpose(0, 1)
    assert not qt.is_contiguous()
    assert torch.allclose(packed_attention(qt, k, v, _cu(lengths)), _per_sequence(q, k, v, lengths), atol=1e-11)


# %%
from torch_judge import evaluate
if not evaluate("0124_packed_attention_eval.py", packed_attention):
    raise SystemExit(1)
