# %%
# 0021 · Clip a group of gradients by their *combined* L2 norm.
# Keep one answer and strengthen it through the three checks.

# %%
import torch

# %%
# Worked probe: [3,4] and [0,12] have a combined norm of 13.
# Clipping to 6.5 multiplies BOTH gradients by 1/2.
g1, g2 = torch.tensor([3., 4.]), torch.tensor([0., 12.])
print("combined norm:", torch.sqrt(g1.square().sum() + g2.square().sum()).item())

# Derive the common scale for the probe gradients at max_norm=6.5.

# %% [answer]
def clip_grad_norm(parameters, max_norm: float) -> float:
    """Scale existing gradients in place; return their original L2 norm."""
    pass

# %%
# Baseline: a sequence of parameters with dense finite gradients, positive
# max_norm; return the original norm as a Python float. No scaling if already
# within the bound. Why would clipping each tensor separately be different?
from torch_judge import check
check("0021_gradient_clipping_eval.py", stage="baseline")

# %%
# Follow-up: accept a one-pass iterable, skip grad=None, and return 0 for
# no gradients. A zero max_norm zeros finite gradients. Parameters repeated in
# the iterable count once; clipping must preserve their combined direction.
check("0021_gradient_clipping_eval.py", stage="followup")

# %%
# Production: reject negative or non-finite max_norm and non-finite gradients
# with ValueError before changing any gradient. Support mixed float32/float64
# dense gradients without concatenating them into one huge tensor. Preserve
# every grad's dtype/device and leave parameter values untouched.
check("0021_gradient_clipping_eval.py", stage="production")
