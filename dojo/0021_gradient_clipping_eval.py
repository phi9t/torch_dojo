"""Reference and cumulative checks for combined L2 gradient clipping."""

# %%
import math
import torch

# %%
def clip_grad_norm(parameters, max_norm: float) -> float:
    try:
        bound = float(max_norm)
    except (TypeError, ValueError) as error:
        raise ValueError("max_norm must be finite and nonnegative") from error
    if not math.isfinite(bound) or bound < 0:
        raise ValueError("max_norm must be finite and nonnegative")
    gradients = []
    seen = set()
    for parameter in parameters:
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        if parameter.grad is not None:
            gradients.append(parameter.grad)
    if not gradients:
        return 0.0
    # Scalar reductions avoid a concatenated gradient allocation and mixed-dtype
    # accumulation avoids overflow in squaring float32 values.
    squared = 0.0
    for grad in gradients:
        if grad.layout != torch.strided or not grad.is_floating_point():
            raise ValueError("need dense floating gradients")
        if not torch.isfinite(grad).all().item():
            raise ValueError("gradient must be finite")
        squared += torch.sum(grad.detach().to(torch.float64).square()).item()
    original = math.sqrt(squared)
    if original > bound:
        scale = bound / original
        with torch.no_grad():
            for grad in gradients:
                grad.mul_(scale)
    return original


# %% [verifiers]
def _parameter_with_grad(values, dtype=torch.float32):
    parameter = torch.nn.Parameter(torch.zeros(len(values), dtype=dtype))
    parameter.grad = torch.tensor(values, dtype=dtype)
    return parameter


def verify_baseline_combined_norm(clip_grad_norm):
    a, b = _parameter_with_grad([3, 4]), _parameter_with_grad([0, 12])
    original = clip_grad_norm([a, b], 6.5)
    assert isinstance(original, float) and abs(original - 13.) < 1e-6
    assert torch.allclose(a.grad, torch.tensor([1.5, 2.]))
    assert torch.allclose(b.grad, torch.tensor([0., 6.]))


def verify_baseline_no_change_below_bound(clip_grad_norm):
    p = _parameter_with_grad([1., 2.])
    before = p.grad.clone()
    assert abs(clip_grad_norm([p], 100) - math.sqrt(5)) < 1e-6
    assert torch.equal(p.grad, before)


def verify_followup_iterator_none_duplicate_zero(clip_grad_norm):
    p = _parameter_with_grad([3., 4.])
    missing = torch.nn.Parameter(torch.zeros(1))
    assert clip_grad_norm(iter([p, missing, p]), 0) == 5.
    assert torch.equal(p.grad, torch.zeros_like(p.grad))
    assert clip_grad_norm(iter([missing]), 1) == 0.


def verify_followup_direction(clip_grad_norm):
    p = _parameter_with_grad([2., -4., 8.])
    before = p.grad.clone()
    clip_grad_norm([p], 2.)
    assert torch.allclose(p.grad / p.grad.norm(), before / before.norm())


def verify_production_mixed_dtype_and_no_parameter_mutation(clip_grad_norm):
    a = _parameter_with_grad([3e20, 4e20], torch.float32)
    b = _parameter_with_grad([12e20], torch.float64)
    old_a, old_b = a.detach().clone(), b.detach().clone()
    original = clip_grad_norm([a, b], 1e20)
    assert math.isclose(original, 13e20, rel_tol=1e-6)
    assert a.grad.dtype == torch.float32 and b.grad.dtype == torch.float64
    assert torch.equal(a.detach(), old_a) and torch.equal(b.detach(), old_b)
    assert math.sqrt(sum(g.double().square().sum().item() for g in (a.grad, b.grad))) <= 1.000001e20


def verify_production_invalid_inputs_do_not_mutate(clip_grad_norm):
    p = _parameter_with_grad([3., 4.])
    for bound in (-1., math.inf, math.nan):
        before = p.grad.clone()
        try:
            clip_grad_norm([p], bound)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted invalid max_norm")
        assert torch.equal(p.grad, before)
    bad = _parameter_with_grad([float("inf")])
    before = p.grad.clone()
    try:
        clip_grad_norm([p, bad], 1.)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted non-finite gradient")
    assert torch.equal(p.grad, before)


# %%
from torch_judge import evaluate
if not evaluate("0021_gradient_clipping_eval.py", clip_grad_norm):
    raise SystemExit(1)
