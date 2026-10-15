# %%
# 0015 · SwiGLU MLP
# Keep one class in the answer cell; revise it as each stage adds a constraint.

# %%
import torch
from torch import nn
from torch.nn import functional as F

# %%
# Baseline — For x = [-2, 0, 2], predict SiLU(x) before running.
# The gate chooses how much of the up projection reaches the down projection.
probe = torch.tensor([-2., 0., 2.])
print("SiLU:", F.silu(probe))
# Implement gate_proj/up_proj: D -> F and down_proj: F -> D, all with bias.
# Return down_proj(silu(gate_proj(x)) * up_proj(x)) for x[..., D].

# %% [answer]
class SwiGLUMLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

# %%
from torch_judge import check
check("0015_mlp_eval.py", stage="baseline")

# %%
# Follow-up — Support x with any leading dimensions, including [N,D] and
# [B,T,D], and non-contiguous inputs. All three projections must remain
# registered module parameters and receive finite gradients.
check("0015_mlp_eval.py", stage="followup")

# %%
# Production — Reject non-positive widths and wrong feature dimension.
# Preserve dtype/device after model.to(...), and handle an empty token axis.
# Parameter count with bias is 2*(D*F+F) + (F*D+D). Derive it for D=3,F=5.
check("0015_mlp_eval.py", stage="production")
