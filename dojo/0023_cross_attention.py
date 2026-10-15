# %%
# 0023 · Multi-head cross-attention
# Edit the answer, then try the three cumulative stages.

# %%
import math
import torch
from torch import nn

# %%
# Baseline: queries x_q [B,Sq,D], keys/values x_kv [B,Sk,D].
# Use Q from x_q and K,V from x_kv, with W_q/k/v/o Linear(D,D).
# Scores are [B,H,Sq,Sk] and the output is [B,Sq,D].
q = torch.randn(1, 2, 4)
kv = torch.randn(1, 5, 4)
print("score shape before heads:", (q @ kv.transpose(-2, -1)).shape)

# Predict the score shape after splitting D=4 into two heads.

# %% [answer]
class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        pass

    def forward(self, x_q, x_kv):
        pass

# %%
from torch_judge import check
check("0023_cross_attention_eval.py", stage="baseline")

# %%
# Follow-up: all encoder positions are visible to every query. Perturb
# the last KV token and inspect the first query; check batch independence.
check("0023_cross_attention_eval.py", stage="followup")

# %%
# Production: validate D/H and input shape, dtype/device; empty queries
# return [B,0,D], while Sk=0 with queries is an error. Preserve gradients.
check("0023_cross_attention_eval.py", stage="production")
