"""Reference and staged checks for flash attention with a recomputing backward."""
# Probe answer: exp(s - L) sums to 1. L = logsumexp(s) is the only per-row number
# the forward must keep: with it, any tile of the probability matrix can be rebuilt
# from Q and K alone, so the [Sq,Sk] matrix never has to be stored.

# %%
import math
import torch


# %%
class FlashAttentionFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, block_size=32):
        if not isinstance(block_size, int) or isinstance(block_size, bool) or block_size <= 0:
            raise ValueError("block_size must be positive int")
        if Q.ndim != 3 or K.ndim != 3 or V.ndim != 3:
            raise ValueError("Q,K,V must be rank three")
        b, sq, d = Q.shape
        if d <= 0 or K.shape[0] != b or K.shape[2] != d or V.shape[:2] != K.shape[:2]:
            raise ValueError("batch, key width and KV length must match")
        if (any(not t.is_floating_point() for t in (Q, K, V))
                or len({t.dtype for t in (Q, K, V)}) != 1
                or len({t.device for t in (Q, K, V)}) != 1):
            raise ValueError("Q,K,V must share floating dtype and device")
        sk = K.shape[1]
        if sk == 0 and sq:
            raise ValueError("queries require keys")
        scale = 1 / math.sqrt(d)
        outs, lses = [], []
        for i in range(0, sq, block_size):
            q = Q[:, i:i + block_size]
            m = Q.new_full((b, q.shape[1], 1), -torch.inf)
            mass = Q.new_zeros((b, q.shape[1], 1))
            acc = Q.new_zeros((b, q.shape[1], V.shape[-1]))
            for j in range(0, sk, block_size):
                s = q @ K[:, j:j + block_size].transpose(-2, -1) * scale
                next_m = torch.maximum(m, s.max(-1, keepdim=True).values)
                correction = torch.exp(m - next_m)
                e = torch.exp(s - next_m)
                acc = acc * correction + e @ V[:, j:j + block_size]
                mass = mass * correction + e.sum(-1, keepdim=True)
                m = next_m
            outs.append(acc / mass)
            lses.append(m + torch.log(mass))
        O = torch.cat(outs, dim=1) if outs else V.new_empty((b, 0, V.shape[-1]))
        L = torch.cat(lses, dim=1) if lses else Q.new_empty((b, 0, 1))
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.block_size = block_size
        return O

    @staticmethod
    def backward(ctx, grad_out):
        Q, K, V, O, L = ctx.saved_tensors
        bs, scale = ctx.block_size, 1 / math.sqrt(Q.shape[-1])
        dQ, dK, dV = torch.zeros_like(Q), torch.zeros_like(K), torch.zeros_like(V)
        delta = (grad_out * O).sum(-1, keepdim=True)          # rowsum(dO * O) = rowsum(P * dP)
        for i in range(0, Q.shape[1], bs):
            q, do, l, dl = Q[:, i:i + bs], grad_out[:, i:i + bs], L[:, i:i + bs], delta[:, i:i + bs]
            for j in range(0, K.shape[1], bs):
                k, v = K[:, j:j + bs], V[:, j:j + bs]
                p = torch.exp(q @ k.transpose(-2, -1) * scale - l)       # rebuilt tile of P
                ds = p * (do @ v.transpose(-2, -1) - dl)                 # softmax backward
                dV[:, j:j + bs] += p.transpose(-2, -1) @ do
                dQ[:, i:i + bs] += ds @ k * scale
                dK[:, j:j + bs] += ds.transpose(-2, -1) @ q * scale
        return dQ, dK, dV, None


# %% [verifiers]
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves


def _dense(Q, K, V):
    return torch.softmax(Q @ K.transpose(-2, -1) / math.sqrt(Q.shape[-1]), -1) @ V


def _reject(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError("invalid input accepted")


class _Peak(TorchDispatchMode):
    """Record the largest tensor any operation produces."""

    def __init__(self):
        super().__init__()
        self.peak = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for t in tree_leaves(out):
            if isinstance(t, torch.Tensor):
                self.peak = max(self.peak, t.numel())
        return out


def _triple(b, sq, sk, d, dv, seed, requires_grad=True):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(b, n, w, dtype=torch.float64, generator=g).requires_grad_(requires_grad)
            for n, w in ((sq, d), (sk, d), (sk, dv))]


def verify_baseline_forward_matches_dense(FlashAttentionFn):
    q, k, v = _triple(2, 8, 8, 4, 3, 1)
    assert torch.allclose(FlashAttentionFn.apply(q, k, v, 3), _dense(q, k, v), atol=1e-11)


def verify_baseline_graph_is_connected(FlashAttentionFn):
    q, k, v = _triple(1, 4, 5, 2, 2, 2)
    out = FlashAttentionFn.apply(q, k, v, 2)
    assert out.requires_grad and out.grad_fn is not None
    out.sum().backward()
    assert all(t.grad is not None for t in (q, k, v))


def verify_followup_backward_matches_dense_autograd(FlashAttentionFn):
    for sq, sk, block in ((5, 7, 1), (5, 7, 3), (6, 6, 4), (3, 9, 100)):
        mine = _triple(2, sq, sk, 4, 3, sq * 10 + sk)
        theirs = [t.detach().clone().requires_grad_() for t in mine]
        w = torch.randn(2, sq, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(9))
        (FlashAttentionFn.apply(*mine, block) * w).sum().backward()
        (_dense(*theirs) * w).sum().backward()
        for a, b in zip(mine, theirs):
            assert torch.allclose(a.grad, b.grad, atol=1e-10), (sq, sk, block)


def verify_followup_large_scores_stay_finite(FlashAttentionFn):
    q = torch.tensor([[[300., 0.], [0., 300.]]], dtype=torch.float64, requires_grad=True)
    k = torch.tensor([[[0., 1.], [1., 0.], [-1., 0.]]], dtype=torch.float64, requires_grad=True)
    v = torch.tensor([[[1.], [3.], [9.]]], dtype=torch.float64, requires_grad=True)
    out = FlashAttentionFn.apply(q, k, v, 1)
    out.sum().backward()
    assert torch.isfinite(out).all() and all(torch.isfinite(t.grad).all() for t in (q, k, v))
    expected = [t.detach().clone().requires_grad_() for t in (q, k, v)]
    _dense(*expected).sum().backward()
    assert all(torch.allclose(a.grad, b.grad, atol=1e-9) for a, b in zip((q, k, v), expected))


def verify_production_gradcheck(FlashAttentionFn):
    q, k, v = _triple(1, 4, 5, 3, 2, 7)
    assert torch.autograd.gradcheck(lambda a, b, c: FlashAttentionFn.apply(a, b, c, 2), (q, k, v))


def verify_production_saves_no_score_matrix(FlashAttentionFn):
    b, s = 2, 64
    q, k, v = _triple(b, s, s, 4, 4, 11)
    saved = []

    def pack(t):
        saved.append(t.numel())
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = FlashAttentionFn.apply(q, k, v, 8)
    assert saved and max(saved) < b * s * s, f"saved tensor of {max(saved)} elements; scores are {b * s * s}"
    out.sum().backward()


def verify_production_no_quadratic_workspace(FlashAttentionFn):
    b, s = 2, 64
    q, k, v = _triple(b, s, s, 4, 4, 12)
    with _Peak() as probe:
        out = FlashAttentionFn.apply(q, k, v, 8)
        out.backward(torch.ones_like(out))
    assert probe.peak < b * s * s, f"an operation produced {probe.peak} elements; scores are {b * s * s}"


def verify_production_strided_inputs_and_upstream(FlashAttentionFn):
    base = _triple(2, 12, 14, 4, 3, 13)
    q, k, v = base[0][:, ::2], base[1][:, ::2], base[2][:, ::2]
    out = FlashAttentionFn.apply(q, k, v, 3)
    assert torch.allclose(out, _dense(q, k, v), atol=1e-11)
    g = torch.randn(3, 6, 2, dtype=torch.float64).permute(2, 1, 0)       # non-contiguous [2,6,3]
    assert not g.is_contiguous()
    out.backward(g)
    ref = [t.detach().clone().requires_grad_() for t in (q, k, v)]
    _dense(*ref).backward(g)
    assert torch.allclose(base[0].grad[:, ::2], ref[0].grad, atol=1e-10)
    assert not base[0].grad[:, 1::2].any()


def verify_production_validation_and_empty(FlashAttentionFn):
    q, k, v = _triple(1, 2, 4, 3, 2, 14, requires_grad=False)
    for bad in (0, -1, 1.5, True):
        _reject(lambda: FlashAttentionFn.apply(q, k, v, bad))
    _reject(lambda: FlashAttentionFn.apply(q, k[:, :0], v[:, :0], 2))
    _reject(lambda: FlashAttentionFn.apply(q.float(), k, v, 2))
    _reject(lambda: FlashAttentionFn.apply(q, k, v[:, :2], 2))
    _reject(lambda: FlashAttentionFn.apply(q[0], k, v, 2))
    empty_q = q[:, :0].clone().requires_grad_()
    out = FlashAttentionFn.apply(empty_q, k, v, 2)
    assert out.shape == (1, 0, 2)
    out.sum().backward()
    assert empty_q.grad.shape == (1, 0, 3)


# %%
from torch_judge import evaluate
if not evaluate("0122_flash_attention_backward_eval.py", FlashAttentionFn):
    raise SystemExit(1)
