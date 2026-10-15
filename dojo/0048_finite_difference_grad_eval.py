"""Reference and cumulative checks for a central-difference gradient checker."""

# %%
import math
from itertools import product
import torch

# %%
def finite_difference_grad(f, x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    if (not isinstance(x, torch.Tensor) or x.dtype != torch.float64
            or not torch.isfinite(x).all().item()):
        raise ValueError("x must be finite float64 tensor")
    try:
        eps = float(eps)
    except (TypeError, ValueError) as error:
        raise ValueError("eps must be finite and positive") from error
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    base = x.detach().clone()
    gradient = torch.empty_like(base)
    with torch.no_grad():
        for position in product(*(range(size) for size in x.shape)):
            step = eps * max(1., abs(base[position].item()))
            plus, minus = base.clone(), base.clone()
            plus[position] += step
            minus[position] -= step
            values = (f(plus), f(minus))
            if any(not isinstance(v, torch.Tensor) or v.numel() != 1
                   or not torch.isfinite(v).all().item() for v in values):
                raise ValueError("f must return a finite scalar tensor")
            gradient[position] = (values[0] - values[1]) / (2 * step)
    return gradient


# %% [verifiers]
def verify_baseline_quadratic_and_linear(finite_difference_grad):
    x = torch.tensor([-2., 0., 3.], dtype=torch.float64)
    grad = finite_difference_grad(lambda z: (z.square() + 2*z).sum(), x)
    assert torch.allclose(grad, torch.tensor([-2., 2., 8.], dtype=x.dtype), atol=1e-8)


def verify_baseline_independent_perturbations(finite_difference_grad):
    x = torch.tensor([1., 2., 3.], dtype=torch.float64)
    seen = []
    def f(z):
        seen.append(z.clone())
        return z.sum()
    assert torch.allclose(finite_difference_grad(f, x), torch.ones_like(x), atol=1e-10)
    assert len(seen) == 2 * x.numel()
    assert torch.equal(x, torch.tensor([1., 2., 3.], dtype=x.dtype))


def verify_followup_scalar_noncontiguous_and_detached(finite_difference_grad):
    scalar = torch.tensor(3., dtype=torch.float64, requires_grad=True)
    g = finite_difference_grad(lambda z: z.square(), scalar)
    assert g.shape == scalar.shape and abs(g.item() - 6) < 1e-8
    x = torch.arange(12., dtype=torch.float64).reshape(3, 4).T.requires_grad_()
    old = x.detach().clone()
    g = finite_difference_grad(lambda z: (z[0, 1] * z[2, 0] + z.square().sum()), x)
    exact = 2 * x.detach().clone()
    exact[0, 1] += x[2, 0].item()
    exact[2, 0] += x[0, 1].item()
    assert g.shape == x.shape and g.dtype == x.dtype and not g.requires_grad
    assert torch.allclose(g, exact, atol=1e-7)
    assert torch.equal(x.detach(), old) and x.grad is None


def verify_production_relative_step_at_large_coordinate(finite_difference_grad):
    x = torch.tensor([1e8], dtype=torch.float64)
    # A fixed 1e-4 perturbation loses digits at this scale.
    g = finite_difference_grad(lambda z: (z / 1e8).square().sum(), x)
    assert torch.allclose(g, torch.tensor([2e-8], dtype=x.dtype), rtol=1e-5, atol=1e-14)


def verify_production_reject_invalid_inputs(finite_difference_grad):
    for bad_eps in (0., -1., float("inf"), float("nan")):
        try:
            finite_difference_grad(lambda z: z.sum(), torch.ones(1, dtype=torch.float64), bad_eps)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted invalid eps")
    for bad_x in (torch.tensor([float("nan")], dtype=torch.float64), torch.ones(1, dtype=torch.float32)):
        try:
            finite_difference_grad(lambda z: z.sum(), bad_x)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted invalid x")
    for bad_f in (lambda z: z, lambda z: torch.tensor(float("inf"))):
        try:
            finite_difference_grad(bad_f, torch.ones(2, dtype=torch.float64))
        except ValueError:
            pass
        else:
            raise AssertionError("accepted nonscalar/nonfinite f")


# %%
from torch_judge import evaluate
if not evaluate("0048_finite_difference_grad_eval.py", finite_difference_grad):
    raise SystemExit(1)
