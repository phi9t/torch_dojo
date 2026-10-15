# %%
# 0010 · Grouped-query attention: one answer through three stages.

# %%
import math
import torch
from torch import nn

# %%
# Baseline: x [B,S,D], H query heads, G KV heads, D divisible by H,
# H divisible by G. Each KV head belongs to H/G adjacent query heads.
# Required projections: W_q [D,D], W_k/W_v [G*(D/H),D], W_o [D,D].
print("KV owner for H=8, G=2:", torch.arange(8) // 4)

# Derive which KV head serves query heads 0, 3, 4, and 7 in the probe.

# %% [answer]
class GroupQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads):
        super().__init__()
        pass

    def forward(self, x, valid_keys=None):
        """valid_keys [B,S] is optional; True marks usable keys."""
        pass

# %%
from torch_judge import check
check("0010_gqa_eval.py", stage="baseline")

# %%
# Follow-up: G=H agrees with ordinary multi-head self-attention when
# projections match. Add optional bool key mask with True=allowed;
# all-masked batches output zeros before W_o (set or omit output bias
# accordingly). Think about the nonzero output projection bias.
check("0010_gqa_eval.py", stage="followup")

# %%
# Production: subclass nn.Module so parameters register; validate
# head divisibility, input shape and mask. Preserve gradients and dtype.
# KV cache width is G*(D/H), rather than D per key/value projection.
check("0010_gqa_eval.py", stage="production")
