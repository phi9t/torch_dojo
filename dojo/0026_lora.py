# %%
# 0026 · Low-rank adaptation of a frozen linear layer
# One editable class, three cumulative stages.

# %%
import torch
from torch import nn
from torch.nn import functional as F

# %%
# Baseline — A has shape [r,in], B is [out,r]. Predict the initial update.
# Freeze nn.Linear(in,out), initialize A randomly and B to exactly zero.
# Forward: linear(x) + alpha/r * (x @ A.T @ B.T).
print("example shapes:", (3, 8), (5, 3), "->", (5, 8))

# %% [answer]
class LoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float = 1.0):
        super().__init__()
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

# %%
from torch_judge import check
check("0026_lora_eval.py", stage="baseline")

# %%
# Follow-up — With B initially zero, A's first gradient is zero but B's
# need not be. After an optimizer step, both can receive gradients. Support
# arbitrary leading dimensions and preserve registered trainable parameters.
check("0026_lora_eval.py", stage="followup")

# %%
# Production — Validate positive widths/rank, finite alpha, and feature width.
# Match the explicit formula after modifying B, including non-contiguous input
# and float64 module conversion. Count trainable parameters: r*(in+out).
check("0026_lora_eval.py", stage="production")
