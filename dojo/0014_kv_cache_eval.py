"""Reference and staged checks for autoregressive KV caching."""
# %%
import math
import torch
from torch import nn

# %%
class KVCacheAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        if (not all(isinstance(n, int) and not isinstance(n, bool) and n > 0 for n in (d_model, num_heads))
                or d_model % num_heads):
            raise ValueError("positive divisible D/H required")
        self.num_heads, self.d_k = num_heads, d_model // num_heads
        self.W_q, self.W_k, self.W_v, self.W_o = (nn.Linear(d_model, d_model) for _ in range(4))

    def forward(self, x, cache=None):
        if x.ndim != 3 or x.shape[-1] != self.W_q.in_features or not x.is_floating_point():
            raise ValueError("x must be floating [B,Snew,D]")
        b, n, d = x.shape
        if cache is not None:
            if (not isinstance(cache, tuple) or len(cache) != 2
                    or any(not isinstance(t, torch.Tensor) or t.ndim != 4 or t.shape[:2] != (b, self.num_heads)
                           or t.shape[-1] != self.d_k or t.dtype != x.dtype or t.device != x.device for t in cache)
                    or cache[0].shape != cache[1].shape):
                raise ValueError("cache must hold matching [B,H,Spast,D/H] tensors")
        q, k, v = [layer(x).reshape(b, n, self.num_heads, self.d_k).transpose(1, 2)
                   for layer in (self.W_q, self.W_k, self.W_v)]
        past = 0 if cache is None else cache[0].shape[2]
        if cache is not None:
            k, v = torch.cat((cache[0], k), dim=2), torch.cat((cache[1], v), dim=2)
        if n == 0:
            return x.new_empty((b, 0, d)), (k, v)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        key_positions = torch.arange(past+n, device=x.device)
        query_positions = past + torch.arange(n, device=x.device)
        future = key_positions[None, :] > query_positions[:, None]
        weights = torch.softmax(scores.masked_fill(future, -torch.inf), dim=-1)
        heads = weights @ v
        out = self.W_o(heads.transpose(1, 2).reshape(b, n, d))
        return out, (k, v)

# %% [verifiers]
def verify_baseline_prefill(KVCacheAttention):
    m = KVCacheAttention(16, 4)
    assert isinstance(m, nn.Module)
    out, (k, v) = m(torch.randn(2, 5, 16))
    assert out.shape == (2, 5, 16) and k.shape == v.shape == (2, 4, 5, 4)


def verify_baseline_single_token_decode(KVCacheAttention):
    torch.manual_seed(1)
    m = KVCacheAttention(8, 2).double()
    x = torch.randn(1, 5, 8, dtype=torch.float64)
    full, _ = m(x)
    pre, cache = m(x[:, :4])
    last, cache = m(x[:, 4:], cache)
    assert torch.allclose(torch.cat((pre, last), 1), full, atol=1e-11)
    assert cache[0].shape[2] == 5


def verify_followup_chunk_equivalence(KVCacheAttention):
    torch.manual_seed(4)
    m = KVCacheAttention(12, 3).double()
    x = torch.randn(2, 9, 12, dtype=torch.float64)
    full, _ = m(x)
    pieces, cache = [], None
    for a, z in ((0, 2), (2, 5), (5, 9)):
        out, cache = m(x[:, a:z], cache)
        pieces.append(out)
    assert torch.allclose(torch.cat(pieces, 1), full, atol=1e-11)
    assert cache[0].shape == (2, 3, 9, 4)


def verify_followup_cache_projection(KVCacheAttention):
    m = KVCacheAttention(8, 2)
    x = torch.randn(1, 3, 8)
    _, cache = m(x)
    expected = m.W_k(x).reshape(1, 3, 2, 4).transpose(1, 2)
    assert torch.allclose(cache[0], expected)


def verify_production_validation(KVCacheAttention):
    for args in ((7, 2), (8, 0), (-8, 2)):
        try:
            KVCacheAttention(*args)
        except ValueError:
            pass
        else:
            raise AssertionError("bad head configuration accepted")
    m = KVCacheAttention(8, 2)
    _, cache = m(torch.randn(1, 2, 8))
    for bad in ((cache[0][:, :, :, :3], cache[1]),
                (cache[0].expand(2, -1, -1, -1), cache[1]),
                (cache[0].double(), cache[1])):
        try:
            m(torch.randn(1, 1, 8), bad)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid cache accepted")
    out, unchanged = m(torch.empty(1, 0, 8), cache)
    assert out.shape == (1, 0, 8) and unchanged[0].shape[2] == 2


def verify_production_gradients(KVCacheAttention):
    m = KVCacheAttention(8, 2).double()
    x = torch.randn(1, 5, 8, dtype=torch.float64, requires_grad=True)
    a, cache = m(x[:, :3])
    b, _ = m(x[:, 3:], cache)
    (a.square().sum() + b.square().sum()).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())

# %%
from torch_judge import evaluate
if not evaluate("0014_kv_cache_eval.py", KVCacheAttention):
    raise SystemExit(1)
