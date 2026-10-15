"""Three-stage exact GELU reference and independent checks."""
# %%
import math
import torch
from torch.nn import functional as F

# %%
def my_gelu(x):
    if not isinstance(x, torch.Tensor) or not x.is_floating_point():
        raise ValueError("floating tensor required")
    return .5 * x * torch.erfc(-x / math.sqrt(2))

# %% [verifiers]
def verify_baseline_known_values(my_gelu):
    x = torch.tensor([-1., 0., 1.])
    out = my_gelu(x)
    assert out.shape == x.shape and out[0] < 0 and out[1] == 0
    assert torch.allclose(out, F.gelu(x), atol=1e-6)


def verify_followup_noncontiguous_float64_gradient(my_gelu):
    x = torch.linspace(-3, 3, 12, dtype=torch.float64).reshape(3, 4).T.requires_grad_()
    out = my_gelu(x)
    assert out.shape == x.shape and out.dtype == x.dtype and out.device == x.device
    assert torch.allclose(out, F.gelu(x), atol=1e-12)
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    zero = torch.tensor([0.], dtype=torch.float64, requires_grad=True)
    my_gelu(zero).backward()
    assert torch.allclose(zero.grad, torch.tensor([.5], dtype=zero.dtype))


def verify_production_negative_tail_and_finiteness(my_gelu):
    x = torch.tensor([-12., -10., -8., 8., 10., 12.], dtype=torch.float64)
    out = my_gelu(x)
    expected = torch.tensor([v*.5*math.erfc(-v/math.sqrt(2)) for v in x.tolist()], dtype=x.dtype)
    assert torch.isfinite(out).all()
    assert torch.allclose(out, expected, rtol=1e-8, atol=0)
    assert torch.all(out[:3] < 0)


def verify_production_reject_nonfloating(my_gelu):
    try:
        my_gelu(torch.tensor([1, 2], dtype=torch.long))
    except ValueError:
        pass
    else:
        raise AssertionError("non-floating input accepted")

# %%
from torch_judge import evaluate
if not evaluate('0019_gelu_eval.py', my_gelu):
    raise SystemExit(1)
