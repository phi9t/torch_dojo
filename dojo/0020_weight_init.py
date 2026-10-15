# %%
# 0020 · Kaiming normal initialization
# Use one editable answer. Each check includes the earlier stages.
# For ReLU layers, Var(W) = 2 / fan_in. Why does a 3×3 convolution have
# nine times the fan_in of a 1×1 convolution with the same input channels?

# %%
import math
import torch

# %%
# Worked probe: 16 input channels, 3×3 kernel -> fan_in = 144.
probe = torch.empty(32, 16, 3, 3)
print("fan_in:", 16 * 3 * 3, "target std:", math.sqrt(2 / 144))

# %% [answer]
def kaiming_init(weight: torch.Tensor) -> torch.Tensor:
    """Fill weight in place with N(0, 2/fan_in) and return that same tensor."""
    pass

# %%
# Baseline: dense [out,in] floating weight. In-place mutation, same object
# returned, standard deviation sqrt(2/in). The random sample will only
# approximate the target moments. Predict whether smaller fan_in widens it.
from torch_judge import check
check("0020_weight_init_eval.py", stage="baseline")

# %%
# Follow-up: convolution [out,in,*kernel] has fan_in=in*product(kernel).
# A one-dimensional bias-like vector uses its length. Every extent must be
# positive. Try a shape [32,16,3,3] beside [32,16,1,1].
check("0020_weight_init_eval.py", stage="followup")

# %%
# Production: preserve dtype/device and caller RNG control; support leaf
# tensors requiring grad by filling under no_grad. Reject empty, non-floating,
# or sparse weights with ValueError. Work in place; avoid an extra full-size
# random tensor. Two fresh tensors after the same manual_seed should agree.
check("0020_weight_init_eval.py", stage="production")
