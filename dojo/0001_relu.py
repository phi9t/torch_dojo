# %%
# 0001 · ReLU: an elementwise activation with a useful derivative
# Revisit the SAME answer at each stage. Run cells individually in Jupyter.
# Rule: do not use torch.relu, F.relu, torch.clamp, or an activation module.

# %%
import torch
from torch_judge import check

# %%
# Baseline: return max(x, 0) elementwise for a floating Tensor. Keep shape and
# dtype. Predict these five outputs before running the probe.
probe = torch.tensor([-2., -1., 0., 1., 2.])
print("positive positions:", probe > 0)
print("expected:", torch.tensor([0., 0., 0., 1., 2.]))

# %% [answer]
def relu(x: torch.Tensor) -> torch.Tensor:
    return x * (x > 0)  # elementwise mask multiplication

# %%
check("0001_relu_eval.py", stage="baseline")

# %%
# Follow-up: handle scalars, higher-rank and noncontiguous views. Preserve
# autograd: the slope is 0 below zero and 1 above zero. At exactly zero use
# the zero subgradient. Do not detach or modify the input.
view = torch.arange(-6., 6.).reshape(3, 4).T
print("view shape/stride:", view.shape, view.stride())
check("0001_relu_eval.py", stage="followup")

# %%
# Production: preserve floating dtype/device, including float64 and CUDA when
# available. No input mutation, even on a leaf requiring gradients. Explain
# why changing dtype in a mask multiplication can violate this contract.
check("0001_relu_eval.py", stage="production")
