# %%
# 0019 · Exact GELU and its negative tail

# %%
import math
import torch

# %%
# Worked probe: GELU(-1) is small and negative, unlike ReLU(-1).
probe = torch.tensor([-1., 0., 1.])
print("Gaussian CDF:", .5 * torch.erfc(-probe / math.sqrt(2)))

# Predict the sign and exact value of GELU(0) before implementing it.

# %% [answer]
def my_gelu(x):
    # Exact x*Phi(x). No built-in GELU; replace this body.
    pass

# %%
# Baseline: use the exact Gaussian-CDF formula, preserving input shape.
from torch_judge import check
check("0019_gelu_eval.py", stage="baseline")

# %%
# Follow-up: support noncontiguous and float64 tensors; preserve
# dtype/device and autograd. At x=0 the derivative is 1/2.
check("0019_gelu_eval.py", stage="followup")

# %%
# Production: avoid cancellation in the negative tail. Prefer
# Phi(x)=erfc(-x/sqrt(2))/2: at x=-8, erf(x/sqrt(2)) rounds close to -1.
# Reject non-floating input and keep output finite over [-12,12].
check("0019_gelu_eval.py", stage="production")
