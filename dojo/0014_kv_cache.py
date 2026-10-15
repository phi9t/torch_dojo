# %%
# 0014 · Autoregressive KV cache
# Edit one answer; check each stage in the same kernel.

# %%
import math
import torch
from torch import nn

# %%
# Baseline: x [B,Snew,D], cache=None or (K,V) each [B,H,Spast,D/H].
# Project only x, append keys and values, attend causally over all keys.
# Output [B,Snew,D] and cache [B,H,Spast+Snew,D/H].
past = torch.zeros(1, 2, 3, 4)
new = torch.ones(1, 2, 2, 4)
print("after two new tokens:", torch.cat([past, new], dim=2).shape)

# Predict the attention key length after appending two tokens to the cache.

# %% [answer]
class KVCacheAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        pass

    def forward(self, x, cache=None):
        pass

# %%
from torch_judge import check
check("0014_kv_cache_eval.py", stage="baseline")

# %%
# Follow-up: prefill may end at any position and decode may append more
# than one token at once. Compare concatenated chunk outputs with a full
# causal pass using the same model. The mask diagonal is Spast+1.
check("0014_kv_cache_eval.py", stage="followup")

# %%
# Production: validate cache shape, batch, dtype and device before concat.
# A zero-length append leaves cache length unchanged. Keep gradients and
# store only projected K/V, not an attention matrix per decode step.
check("0014_kv_cache_eval.py", stage="production")
