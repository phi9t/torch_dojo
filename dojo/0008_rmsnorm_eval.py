"""Three-stage RMSNorm reference and independent checks."""
# %%
import torch

# %%
def rms_norm(x, weight, eps=1e-6):
    if not isinstance(x, torch.Tensor) or not isinstance(weight, torch.Tensor) or x.ndim < 1 or x.shape[-1] < 1 or weight.shape != (x.shape[-1],):
        raise ValueError("x [...,D] and weight [D] required")
    if not x.is_floating_point() or not weight.is_floating_point() or x.dtype != weight.dtype or x.device != weight.device or not (0 < eps < float('inf')):
        raise ValueError("matching floating dtype/device and eps>0 required")
    work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    factor = torch.rsqrt(work.square().mean(dim=-1, keepdim=True) + eps)
    return (work * factor).to(x.dtype) * weight

# %% [verifiers]
def verify_baseline_known_vector(rms_norm):
    x, w = torch.tensor([[3., 4.]]), torch.tensor([2., .5])
    expected = x / (torch.tensor(12.5 + 1e-6).sqrt()) * w
    assert torch.allclose(rms_norm(x, w), expected, atol=1e-6)


def verify_followup_high_rank_noncontiguous_and_grad(rms_norm):
    x = torch.randn(2, 4, 3, dtype=torch.float64).transpose(1, 2).requires_grad_()
    w = torch.linspace(.5, 2., 4, dtype=x.dtype, requires_grad=True)
    out = rms_norm(x, w)
    expected = x / torch.sqrt(x.square().mean(-1, keepdim=True) + 1e-6) * w
    assert out.shape == x.shape and torch.allclose(out, expected, atol=1e-12)
    out.square().sum().backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in (x, w))
    assert torch.equal(rms_norm(torch.zeros(1, 4), torch.ones(4)), torch.zeros(1, 4))


def verify_production_mixed_precision_no_square_overflow(rms_norm):
    x = torch.tensor([[300., -400., 500.]], dtype=torch.float16, requires_grad=True)
    w = torch.ones(3, dtype=x.dtype, requires_grad=True)
    out = rms_norm(x, w)
    expected = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)).half()
    assert out.dtype == x.dtype and torch.allclose(out, expected, atol=1e-3)
    out.sum().backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(w.grad).all()


def verify_production_invalid_contract(rms_norm):
    for x, w, eps in ((torch.ones(2, 3), torch.ones(2), 1e-6), (torch.ones(2, 3), torch.ones(3), 0), (torch.ones(2, 3, dtype=torch.long), torch.ones(3), 1e-6)):
        try:
            rms_norm(x, w, eps)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid RMSNorm contract accepted")

# %%
from torch_judge import evaluate
if not evaluate('0008_rmsnorm_eval.py', rms_norm):
    raise SystemExit(1)
