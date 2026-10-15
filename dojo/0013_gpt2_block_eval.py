"""Reference and staged checks for a pre-norm causal transformer block."""
# %%
import math
import torch
from torch import nn

# %%
class GPT2Block(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        if (not all(isinstance(n, int) and not isinstance(n, bool) and n > 0 for n in (d_model, num_heads))
                or d_model % num_heads):
            raise ValueError("positive D and H with D divisible by H required")
        self.num_heads, self.d_k = num_heads, d_model // num_heads
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.W_q, self.W_k, self.W_v, self.W_o = (nn.Linear(d_model, d_model) for _ in range(4))
        self.mlp = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.GELU(), nn.Linear(4*d_model, d_model))

    def forward(self, x):
        if x.ndim != 3 or x.shape[-1] != self.W_q.in_features or not x.is_floating_point():
            raise ValueError("x must be floating [B,S,D]")
        b, s, d = x.shape
        if s == 0:
            return x.clone()
        y = self.ln1(x)
        q, k, v = [layer(y).reshape(b, s, self.num_heads, self.d_k).transpose(1, 2)
                   for layer in (self.W_q, self.W_k, self.W_v)]
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        future = torch.ones((s, s), dtype=torch.bool, device=x.device).triu(1)
        weights = torch.softmax(scores.masked_fill(future, -torch.inf), dim=-1)
        heads = weights @ v
        y = x + self.W_o(heads.transpose(1, 2).reshape(b, s, d))
        return y + self.mlp(self.ln2(y))

# %% [verifiers]
def verify_baseline_architecture(GPT2Block):
    m = GPT2Block(16, 4)
    assert isinstance(m, nn.Module)
    assert isinstance(m.ln1, nn.LayerNorm) and isinstance(m.ln2, nn.LayerNorm)
    assert all(getattr(m, name).weight.shape == (16, 16) for name in ("W_q", "W_k", "W_v", "W_o"))
    layers = list(m.mlp)
    assert layers[0].weight.shape == (64, 16) and isinstance(layers[1], nn.GELU)
    assert layers[2].weight.shape == (16, 64)
    assert m(torch.randn(2, 5, 16)).shape == (2, 5, 16)


def verify_baseline_residual_path(GPT2Block):
    m = GPT2Block(8, 2)
    with torch.no_grad():
        for p in m.parameters():
            p.zero_()
    x = torch.randn(1, 3, 8)
    assert torch.equal(m(x), x)


def verify_followup_future_invariance(GPT2Block):
    torch.manual_seed(7)
    m = GPT2Block(12, 3).double()
    x = torch.randn(2, 7, 12, dtype=torch.float64)
    altered = x.clone()
    altered[:, 4:] += 100
    assert torch.allclose(m(x)[:, :4], m(altered)[:, :4], atol=1e-11)


def verify_followup_batch_isolation(GPT2Block):
    m = GPT2Block(8, 2)
    x = torch.randn(2, 4, 8)
    first = m(x)[0]
    x[1] *= 200
    assert torch.allclose(first, m(x)[0])


def verify_production_validation(GPT2Block):
    for args in ((0, 2), (8, 0), (7, 2), (8, -2)):
        try:
            GPT2Block(*args)
        except ValueError:
            pass
        else:
            raise AssertionError("bad head configuration accepted")
    m = GPT2Block(8, 2)
    for bad in (torch.randn(8), torch.randn(1, 2, 7)):
        try:
            m(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("bad input accepted")
    assert m(torch.empty(2, 0, 8)).shape == (2, 0, 8)


def verify_production_gradients(GPT2Block):
    m = GPT2Block(8, 2).double()
    x = torch.randn(2, 4, 8, dtype=torch.float64, requires_grad=True)
    m(x).square().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())

# %%
from torch_judge import evaluate
if not evaluate("0013_gpt2_block_eval.py", GPT2Block):
    raise SystemExit(1)
