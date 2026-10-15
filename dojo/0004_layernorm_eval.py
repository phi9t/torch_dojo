"""Reference and staged checks for final-axis LayerNorm."""

# %%
import math
import torch

# %%
def my_layer_norm(x, gamma, beta, eps=1e-5):
    if (not isinstance(x, torch.Tensor) or not isinstance(gamma, torch.Tensor)
            or not isinstance(beta, torch.Tensor)):
        raise ValueError("x, gamma, and beta must be tensors")
    if (not x.is_floating_point() or not gamma.is_floating_point()
            or not beta.is_floating_point() or x.ndim < 1 or x.shape[-1] == 0
            or gamma.shape != (x.shape[-1],) or beta.shape != (x.shape[-1],)
            or x.dtype != gamma.dtype or x.dtype != beta.dtype
            or x.device != gamma.device or x.device != beta.device
            or not isinstance(eps, (int, float)) or not math.isfinite(eps) or eps <= 0):
        raise ValueError("need floating tensors on one dtype/device with matching feature size and eps>0")
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    return gamma * (x - mean) / torch.sqrt(var + eps) + beta

# %% [verifiers]
def verify_baseline_known_values(my_layer_norm):
    x = torch.tensor([[1., 3.], [5., 7.]])
    y = my_layer_norm(x, torch.ones(2), torch.zeros(2), eps=1e-5)
    expected = torch.tensor([[-1., 1.], [-1., 1.]]) / (1. + 1e-5) ** .5
    assert y.shape == x.shape and torch.allclose(y, expected, atol=1e-6)


def verify_baseline_affine(my_layer_norm):
    x = torch.randn(2, 3, 4)
    gamma, beta = torch.tensor([1., 2., 3., 4.]), torch.tensor([-1., 0., 1., 2.])
    assert torch.allclose(my_layer_norm(x, gamma, beta), torch.nn.functional.layer_norm(x, (4,), gamma, beta), atol=1e-5)


def verify_followup_constant_and_singleton(my_layer_norm):
    x = torch.tensor([[[3.], [5.]]], dtype=torch.float64)
    gamma, beta = torch.tensor([2.], dtype=x.dtype), torch.tensor([-1.], dtype=x.dtype)
    y = my_layer_norm(x, gamma, beta)
    assert torch.equal(y, torch.full_like(x, -1.))
    assert torch.isfinite(y).all()


def verify_followup_noncontiguous(my_layer_norm):
    x = torch.randn(2, 4, 3, dtype=torch.float64).transpose(0, 1)
    gamma, beta = torch.randn(3, dtype=x.dtype), torch.randn(3, dtype=x.dtype)
    y = my_layer_norm(x, gamma, beta)
    ref = torch.nn.functional.layer_norm(x, (3,), gamma, beta)
    assert y.shape == x.shape and torch.allclose(y, ref, atol=1e-12, rtol=1e-12)


def verify_production_backward_and_immutability(my_layer_norm):
    for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
        x = torch.randn(2, 3, 5, dtype=torch.float64, device=device, requires_grad=True)
        gamma = torch.randn(5, dtype=x.dtype, device=device, requires_grad=True)
        beta = torch.randn(5, dtype=x.dtype, device=device, requires_grad=True)
        originals = [t.detach().clone() for t in (x, gamma, beta)]
        ref_inputs = [t.detach().clone().requires_grad_() for t in (x, gamma, beta)]
        y = my_layer_norm(x, gamma, beta)
        ref = torch.nn.functional.layer_norm(ref_inputs[0], (5,), ref_inputs[1], ref_inputs[2])
        weights = torch.randn_like(y)
        (y * weights).sum().backward()
        (ref * weights).sum().backward()
        assert y.shape == x.shape and y.dtype == x.dtype and y.device == x.device
        assert torch.allclose(y, ref, atol=1e-12, rtol=1e-12)
        for actual, baseline, original in zip((x, gamma, beta), ref_inputs, originals):
            assert torch.allclose(actual.grad, baseline.grad, atol=1e-11, rtol=1e-11)
            assert torch.equal(actual.detach(), original)


def verify_production_validation(my_layer_norm):
    x = torch.ones(2, 3)
    for bad_gamma, bad_beta, eps in ((torch.ones(4), torch.zeros(3), 1e-5),
                                     (torch.ones(3), torch.zeros(2), 1e-5),
                                     (torch.ones(3), torch.zeros(3), 0.),
                                     (torch.ones(3), torch.zeros(3), -1.),
                                     (torch.ones(3), torch.zeros(3), float("nan"))):
        try:
            my_layer_norm(x, bad_gamma, bad_beta, eps)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted invalid shape or epsilon")
    try:
        my_layer_norm(torch.ones(2, 3, dtype=torch.long), torch.ones(3), torch.zeros(3))
    except ValueError:
        pass
    else:
        raise AssertionError("accepted nonfloating input")


# %%
from torch_judge import evaluate
if not evaluate("0004_layernorm_eval.py", my_layer_norm):
    raise SystemExit(1)
