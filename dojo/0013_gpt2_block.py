# %%
# 0013 · GPT-2 style pre-norm block
# Edit one answer and advance through three cumulative checks.

# %%
import math
import torch
from torch import nn

# %%
# Baseline: x [B,S,D] -> x + causal_attention(ln1(x)) ->
# y + mlp(ln2(y)). H divides D. Attention has W_q/k/v/o; MLP is
# Linear(D,4D), GELU, Linear(4D,D). Predict the residual shape.
x = torch.randn(2, 3, 8)
print("residual shape:", (x + torch.zeros_like(x)).shape)

# %% [answer]
class GPT2Block(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        pass

    def forward(self, x):
        pass

# %%
from torch_judge import check
check("0013_gpt2_block_eval.py", stage="baseline")

# %%
# Follow-up: perturb only future tokens and verify earlier outputs cannot
# change. The mask's diagonal remains visible; batches stay independent.
check("0013_gpt2_block_eval.py", stage="followup")

# %%
# Production: validate positive divisible head widths and input [B,S,D],
# accept S=0, preserve dtype/device and finite gradients to every parameter.
check("0013_gpt2_block_eval.py", stage="production")
