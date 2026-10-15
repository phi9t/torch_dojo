# %%
# 0009 · Causal self-attention: one editable answer, three stages.

# %%
import math
import torch

# %%
# Baseline: Q,K [B,S,D], V [B,S,Dv] -> [B,S,Dv].
# Query i can attend only to keys 0..i. Predict the output at i=0.
positions = torch.arange(4)
print("future keys:", positions[None, :] > positions[:, None])

# %% [answer]
def causal_attention(Q, K, V, valid_keys=None):
    """valid_keys [B,S] is optional; True means that key is present."""
    pass

# %%
from torch_judge import check
check("0009_causal_attention_eval.py", stage="baseline")

# %%
# Follow-up: optional bool valid_keys [B,S]. Combine with causal mask.
# A query with no valid keys outputs zero; no NaN. Mutating future K/V
# must not affect earlier outputs. What if key 0 is invalid?
check("0009_causal_attention_eval.py", stage="followup")

# %%
# Production: validate shapes, dtype and mask; handle S=0, backward,
# non-contiguous input and batch isolation. Dense scores cost O(B*S*S).
check("0009_causal_attention_eval.py", stage="production")
