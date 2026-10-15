# %%
# 0048 · Central finite-difference gradient
# Adapted from ferric_continuum/tnsr's autograd gradient checker.
# One editable function: revise it as the contracts grow.

# %%
import torch

# %%
# Worked probe: f(x)=sum(x²), x=[-2,0,3]. Predict df/dx first.
x = torch.tensor([-2., 0., 3.], dtype=torch.float64)
f = lambda z: z.square().sum()
step = 1e-4
plus, minus = x.clone(), x.clone()
plus[0] += step
minus[0] -= step
print("first partial:", ((f(plus) - f(minus)) / (2 * step)).item())

# %% [answer]
def finite_difference_grad(f, x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Return central differences with x's shape/dtype, detached from autograd."""
    pass

# %%
# Baseline: finite float64 vector x and scalar f(x). Evaluate f at independent
# plus/minus copies for each coordinate. No backward() or autograd.grad().
from torch_judge import check
check("0048_finite_difference_grad_eval.py", stage="baseline")

# %%
# Follow-up: arbitrary tensor shape, including scalar and noncontiguous views.
# Leave x and x.grad untouched, including when x.requires_grad is true.
# Test a coupled function where one coordinate affects several terms.
check("0048_finite_difference_grad_eval.py", stage="followup")

# %%
# Production: eps must be finite and positive; x must be finite float64 and
# f must return one finite scalar for every probe. Raise ValueError otherwise.
# Use h_i=eps*max(1,abs(x_i)) to reduce poor relative steps at large x.
# Output is detached, same shape/dtype/device. Use O(x.numel) f calls and
# bounded scratch space; stay away from non-differentiable kinks.
check("0048_finite_difference_grad_eval.py", stage="production")
