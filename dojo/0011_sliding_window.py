# %%
# 0011 · Sliding-window attention: one answer through three stages.

# %%
import math
import torch

# %%
# Baseline: Q,K [B,S,D], V [B,S,Dv]; integer radius w>=0.
# Query i sees keys j with |i-j|<=w, inclusive. No attention helper.
# Predict the visible keys for i=2,w=1 from this distance matrix.
pos = torch.arange(5)
print("distance:", (pos[:, None] - pos[None, :]).abs())

# %% [answer]
def sliding_window_attention(Q, K, V, window_size, valid_keys=None):
    """valid_keys [B,S] is optional; True means that key is present."""
    pass

# %%
from torch_judge import check
check("0011_sliding_window_eval.py", stage="baseline")

# %%
# Follow-up: boundary windows shrink rather than wrap. Optional bool
# valid_keys [B,S] uses True=allowed. All-masked windows output zero.
# Window=0 returns V if every key is valid; window>=S is full attention.
check("0011_sliding_window_eval.py", stage="followup")

# %%
# Production: validate inputs and allow S=0. Support backward, batch
# isolation and non-contiguous tensors. Avoid a dense [B,S,S] score tensor:
# keep only O(B*S*(2w+1)) score storage, or less, for w much smaller than S.
check("0011_sliding_window_eval.py", stage="production")
