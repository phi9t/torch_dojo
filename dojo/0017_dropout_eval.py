"""Three-stage inverted dropout reference and checks."""
# %%
import math
import torch
import torch.nn as nn

# %%
class MyDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        if not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("p must be finite and in [0,1]")
        self.p = float(p)

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        if self.p == 1:
            return x * 0
        mask = (torch.rand(x.shape, device=x.device) >= self.p).to(x.dtype)
        return x * mask / (1 - self.p)

# %% [verifiers]
def verify_baseline_mask_and_scale(MyDropout):
    torch.manual_seed(42)
    layer = MyDropout(.5)
    layer.train()
    out = layer(torch.ones(1000))
    assert isinstance(layer, nn.Module) and (out == 0).any() and (out != 0).any()
    assert torch.all(out[(out != 0)] == 2)
    assert .43 < (out == 0).float().mean().item() < .57


def verify_followup_eval_and_endpoints(MyDropout):
    x = torch.tensor([1., 2.], requires_grad=True)
    layer = MyDropout(.3).eval()
    assert torch.equal(layer(x), x)
    assert torch.equal(MyDropout(0)(x), x)
    zero = MyDropout(1)(x)
    assert torch.equal(zero, torch.zeros_like(x))
    zero.sum().backward()
    assert torch.equal(x.grad, torch.zeros_like(x))


def verify_production_seed_dtype_gradient_and_immutability(MyDropout):
    x = torch.ones(32, dtype=torch.float64, requires_grad=True)
    before = x.detach().clone()
    layer = MyDropout(.25).train()
    torch.manual_seed(123)
    a = layer(x)
    torch.manual_seed(123)
    b = layer(x)
    assert a.dtype == x.dtype and a.device == x.device and a.shape == x.shape
    assert torch.equal(a, b) and torch.equal(x.detach(), before)
    a.sum().backward()
    assert torch.allclose(x.grad, a.detach(), atol=1e-12)


def verify_production_reject_invalid_probability(MyDropout):
    for p in (-.1, 1.1, float('nan'), float('inf')):
        try:
            MyDropout(p)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid dropout probability accepted")

# %%
from torch_judge import evaluate
if not evaluate('0017_dropout_eval.py', MyDropout):
    raise SystemExit(1)
