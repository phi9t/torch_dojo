# %%
# 0003 · SimpleLinear: a trainable affine map
# Keep one answer class and revise it through the three stages.
# Do not use torch.nn.Linear in the answer.

# %%
import math
import torch
from torch_judge import check

# %%
# Baseline: positive in_features/out_features. Create leaf weight [O,I]
# from randn / sqrt(I), leaf zero bias [O], both requiring gradients.
# forward(x) = x @ weight.T + bias for x [N,I].
x = torch.ones(2, 3)
w = torch.ones(4, 3)
print("matrix product shape:", (x @ w.T).shape)

# Predict the output shape and parameter count for I=3, O=4 before coding.

# %% [answer]
class SimpleLinear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, *, dtype=None, device=None):
        for n in (in_features, out_features):
            if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
                raise ValueError("in_features and out_features must be positive integers.")
        super().__init__()
        # Parameter registers with the Module, so .parameters(), .to(), and
        # state_dict() see them; a bare tensor attribute would be invisible.
        self.weight = torch.nn.Parameter(
            torch.randn(out_features, in_features, dtype=dtype, device=device) / math.sqrt(in_features))
        self.bias = torch.nn.Parameter(torch.zeros(out_features, dtype=dtype, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 1 or x.shape[-1] != self.weight.shape[1]:
            raise ValueError(f"Input width must equal in_features {self.weight.shape[1]}, got shape {tuple(x.shape)}.")
        return x @ self.weight.T + self.bias

# %%
check("0003_linear_eval.py", stage="baseline")

# %%
# Follow-up: allow a single vector [I] and arbitrary batch prefix [..., I],
# including noncontiguous views. Gradients must reach input, weight, and bias.
# Adding the bias should broadcast only over the final feature dimension.
print("example prefix:", torch.empty(2, 3, 4).shape[:-1])
check("0003_linear_eval.py", stage="followup")

# %%
# Production: reject nonpositive feature counts and wrong input width.
# Preserve input values. Optional keyword-only dtype/device constructor args
# allow a caller to create the parameters directly where computation occurs;
# default behavior remains float32 on CPU. No hidden reinitialization in forward.
check("0003_linear_eval.py", stage="production")
