"""Staged reference and verifiers for grouped-query attention."""

# %%
import math
import torch
from torch import nn

# %%
class GroupQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads):
        super().__init__()
        if (not all(isinstance(x, int) and not isinstance(x, bool) and x > 0
                    for x in (d_model, num_heads, num_kv_heads))
                or d_model % num_heads or num_heads % num_kv_heads):
            raise ValueError("require positive D,H,G with D%H=H%G=0")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, num_kv_heads * self.d_k)
        self.W_v = nn.Linear(d_model, num_kv_heads * self.d_k)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, valid_keys=None):
        if x.ndim != 3 or x.shape[-1] != self.W_q.in_features or not x.is_floating_point():
            raise ValueError("x must be floating [B,S,D]")
        b, s, _ = x.shape
        if valid_keys is not None and (valid_keys.shape != (b, s) or valid_keys.dtype != torch.bool or valid_keys.device != x.device):
            raise ValueError("valid_keys must be bool [B,S]")
        q = self.W_q(x).reshape(b, s, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).reshape(b, s, self.num_kv_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).reshape(b, s, self.num_kv_heads, self.d_k).transpose(1, 2)
        repeats = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)
        if s == 0:
            return self.W_o(x.new_empty((b, 0, self.W_q.in_features)))
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if valid_keys is None:
            weights = torch.softmax(scores, dim=-1)
        else:
            allowed = valid_keys[:, None, None, :]
            active = allowed.any(dim=-1, keepdim=True)
            masked = scores.masked_fill(~allowed, -torch.inf)
            weights = torch.softmax(torch.where(active, masked, torch.zeros_like(masked)), dim=-1)
            weights = weights * allowed.to(weights.dtype)
        heads = torch.matmul(weights, v)
        combined = heads.transpose(1, 2).contiguous().reshape(b, s, -1)
        output = self.W_o(combined)
        if valid_keys is not None:
            output = output * valid_keys.any(dim=1)[:, None, None].to(output.dtype)
        return output

# %% [verifiers]
def verify_baseline_projection_contract(GroupQueryAttention):
    gqa = GroupQueryAttention(16, 4, 2)
    assert gqa.W_q.weight.shape == (16, 16)
    assert gqa.W_k.weight.shape == (8, 16)
    assert gqa.W_v.weight.shape == (8, 16)
    assert gqa.W_o.weight.shape == (16, 16)
    assert gqa.forward(torch.randn(2, 5, 16)).shape == (2, 5, 16)


def verify_baseline_grouped_numerics(GroupQueryAttention):
    torch.manual_seed(10)
    model = GroupQueryAttention(8, 4, 2).double()
    x = torch.randn(2, 3, 8, dtype=torch.float64)
    q = model.W_q(x).reshape(2, 3, 4, 2).transpose(1, 2)
    k = model.W_k(x).reshape(2, 3, 2, 2).transpose(1, 2).repeat_interleave(2, 1)
    v = model.W_v(x).reshape(2, 3, 2, 2).transpose(1, 2).repeat_interleave(2, 1)
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0)
    expected = model.W_o(expected.transpose(1, 2).reshape(2, 3, 8))
    assert torch.allclose(model.forward(x), expected, atol=1e-12)


def verify_followup_full_heads_and_mask(GroupQueryAttention):
    model = GroupQueryAttention(8, 4, 4).double()
    x = torch.randn(2, 3, 8, dtype=torch.float64)
    assert model.W_k.out_features == 8
    mask = torch.tensor([[True, False, False], [False, False, False]])
    got = model.forward(x, mask)
    assert torch.equal(got[1], torch.zeros_like(got[1]))
    v = model.W_v(x[:1, :1])
    expected = model.W_o(v)
    assert torch.allclose(got[0], expected.expand_as(got[:1])[0], atol=1e-12)


def verify_followup_batch_isolation(GroupQueryAttention):
    torch.manual_seed(3)
    model = GroupQueryAttention(8, 4, 2)
    x = torch.randn(2, 4, 8)
    first = model.forward(x)[0]
    x[1] *= 100
    assert torch.allclose(first, model.forward(x)[0])


def verify_production_module_gradients(GroupQueryAttention):
    model = GroupQueryAttention(8, 4, 2).double()
    assert isinstance(model, nn.Module)
    assert len(list(model.parameters())) == 8
    x = torch.randn(2, 4, 8, dtype=torch.float64, requires_grad=True)
    model.forward(x).square().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def verify_production_validation(GroupQueryAttention):
    for spec in ((7, 4, 2), (8, 3, 1), (8, 4, 3), (0, 4, 2)):
        try:
            GroupQueryAttention(*spec)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid head contract accepted")
    model = GroupQueryAttention(8, 4, 2)
    try:
        model.forward(torch.randn(1, 2, 8), torch.ones(1, 3, dtype=torch.bool))
    except ValueError:
        pass
    else:
        raise AssertionError("invalid mask accepted")

# %%
from torch_judge import evaluate
if not evaluate("0010_gqa_eval.py", GroupQueryAttention):
    raise SystemExit(1)
