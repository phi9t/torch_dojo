"""Reference and staged checks for multi-head cross-attention."""
# %%
import math
import torch
from torch import nn

# %%
class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        if (not all(isinstance(n, int) and not isinstance(n, bool) and n > 0 for n in (d_model, num_heads))
                or d_model % num_heads):
            raise ValueError("positive divisible D/H required")
        self.num_heads, self.d_k = num_heads, d_model // num_heads
        self.W_q, self.W_k, self.W_v, self.W_o = (nn.Linear(d_model, d_model) for _ in range(4))

    def forward(self, x_q, x_kv):
        if (x_q.ndim != 3 or x_kv.ndim != 3 or x_q.shape[0] != x_kv.shape[0]
                or x_q.shape[-1] != self.W_q.in_features or x_kv.shape[-1] != self.W_q.in_features
                or not x_q.is_floating_point() or x_q.dtype != x_kv.dtype or x_q.device != x_kv.device):
            raise ValueError("query and KV must share floating [B,S,D] contract")
        b, sq, d = x_q.shape
        sk = x_kv.shape[1]
        if sk == 0 and sq:
            raise ValueError("queries need at least one key")
        if sq == 0:
            return x_q.clone()
        q = self.W_q(x_q).reshape(b, sq, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x_kv).reshape(b, sk, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x_kv).reshape(b, sk, self.num_heads, self.d_k).transpose(1, 2)
        weights = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.d_k), dim=-1)
        return self.W_o((weights @ v).transpose(1, 2).reshape(b, sq, d))

# %% [verifiers]
def verify_baseline_lengths(MultiHeadCrossAttention):
    m = MultiHeadCrossAttention(16, 4)
    assert isinstance(m, nn.Module)
    assert m(torch.randn(2, 3, 16), torch.randn(2, 7, 16)).shape == (2, 3, 16)


def verify_baseline_single_key(MultiHeadCrossAttention):
    m = MultiHeadCrossAttention(8, 2).double()
    q, kv = torch.randn(1, 4, 8, dtype=torch.float64), torch.randn(1, 1, 8, dtype=torch.float64)
    expected = m.W_o(m.W_v(kv)).expand(1, 4, 8)
    assert torch.allclose(m(q, kv), expected, atol=1e-11)


def verify_followup_all_keys_visible(MultiHeadCrossAttention):
    m = MultiHeadCrossAttention(8, 2)
    q, kv = torch.randn(1, 3, 8), torch.randn(1, 5, 8)
    changed = kv.clone(); changed[:, -1] += 50
    assert not torch.allclose(m(q, kv)[:, 0], m(q, changed)[:, 0])


def verify_followup_batch_independence(MultiHeadCrossAttention):
    m = MultiHeadCrossAttention(8, 2)
    q, kv = torch.randn(2, 3, 8), torch.randn(2, 5, 8)
    first = m(q, kv)[0]
    kv[1] *= 100
    assert torch.allclose(first, m(q, kv)[0])


def verify_production_contract(MultiHeadCrossAttention):
    for args in ((7, 2), (8, 0), (-8, 2)):
        try:
            MultiHeadCrossAttention(*args)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid heads accepted")
    m = MultiHeadCrossAttention(8, 2)
    assert m(torch.empty(2, 0, 8), torch.randn(2, 3, 8)).shape == (2, 0, 8)
    for q, kv in ((torch.randn(1, 2, 8), torch.empty(1, 0, 8)),
                  (torch.randn(1, 2, 8), torch.randn(2, 3, 8)),
                  (torch.randn(1, 2, 8).double(), torch.randn(1, 3, 8))):
        try:
            m(q, kv)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid input accepted")


def verify_production_gradients(MultiHeadCrossAttention):
    m = MultiHeadCrossAttention(8, 2).double()
    q = torch.randn(1, 3, 8, dtype=torch.float64, requires_grad=True)
    kv = torch.randn(1, 5, 8, dtype=torch.float64, requires_grad=True)
    m(q, kv).square().sum().backward()
    assert q.grad is not None and kv.grad is not None
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())

# %%
from torch_judge import evaluate
if not evaluate("0023_cross_attention_eval.py", MultiHeadCrossAttention):
    raise SystemExit(1)
