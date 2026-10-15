# %%
# 0123 · Split-KV attention (flash-decoding)
# Decode has one query token and a long cache, so parallelism has to come from
# the keys. Each split attends to its own slice and reports (output, logsumexp);
# a cheap merge combines them. Authored; see 0014 (KV cache) and 0025 (tiling).

# %%
import math
import torch
from torch_judge import check

# %%
# Baseline — two halves of one softmax. Predict lse for the first half, then
# whether weighting each half's output by exp(lse_i - lse_total) is exact.
s = torch.tensor([1., 2., 3., 4.])
half_lse = torch.logsumexp(s[:2], dim=0)
print(half_lse.item(), torch.logsumexp(s, dim=0).item())

# %% [drill]
# Each half's output is normalised by exp(half_lse), not by the full mass.
# What factor undoes that normalisation, and what is it relative to the total?

# %%
# Contract: split_kv_attention(Q, K, V, num_splits, allowed=None).
# Q [B,Tq,D], K [B,Tk,D], V [B,Tk,Dv] -> [B,Tq,Dv]. Cut the Tk keys into
# num_splits contiguous slices with edges i*Tk//num_splits (slices may be empty
# when num_splits > Tk; skip them). For each slice compute its scaled scores,
# its normalised output and its per-row logsumexp, then merge with
# weights exp(lse_i - lse_total). Never form scores for all Tk keys at once.

# %%
# Inputs conforming to the signature below; a decode-like shape with an uneven split.
torch.manual_seed(0)
Q: torch.Tensor = torch.randn(2, 1, 4, dtype=torch.float64)       # [B,Tq,D]
K: torch.Tensor = torch.randn(2, 10, 4, dtype=torch.float64)      # [B,Tk,D]
V: torch.Tensor = torch.randn(2, 10, 3, dtype=torch.float64)      # [B,Tk,Dv]
num_splits: int = 3                                               # > 0, may exceed Tk
allowed: torch.Tensor | None = None                               # bool, broadcastable to [B,Tq,Tk]


# %% [answer]
def split_kv_attention(
    Q: torch.Tensor,                        # [B,Tq,D]
    K: torch.Tensor,                        # [B,Tk,D]
    V: torch.Tensor,                        # [B,Tk,Dv]
    num_splits: int,
    allowed: torch.Tensor | None = None,    # bool, broadcastable to [B,Tq,Tk]
) -> torch.Tensor:                          # [B,Tq,Dv]
    pass

# %%
# Baseline: matches dense attention for any split count, including a decode step
# with Tq=1 and a cache that does not divide evenly.
check("0123_split_kv_attention_eval.py", stage="baseline")

# %%
# Follow-up: allowed is an optional bool mask broadcastable to [B,Tq,Tk]
# (True = attend). A split can be fully masked for a row, so its lse is -inf
# and its weight must be exactly 0 without producing NaN. A row with no allowed
# key at all returns exactly zero. More splits than keys works.
check("0123_split_kv_attention_eval.py", stage="followup")

# %%
# Production: scores of 500 stay finite; gradients match dense attention even
# through fully masked splits and rows (think about where -inf minus -inf and
# log(0) hide); the largest tensor any operation makes is under half the full
# [Tq,Tk] score matrix when Tk is split 16 ways; validate num_splits, shapes,
# dtype, device and mask with ValueError; non-contiguous inputs work.
check("0123_split_kv_attention_eval.py", stage="production")
