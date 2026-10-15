# %%
# 0114 · Collisions are contributions
# Goal: add K values into `size` slots when several values may share a slot.
# Fundamentals behind 0107 (voxel pooling) and 0108 (BEV scatter). Authored
# practice; the contracts below are authored, not imported task text.

# %%
import torch
from torch_judge import check

# %%
# Baseline — Three values go to slots [0, 0, 2] of a length-3 output.
# Predict: what should slot 0 hold if both values count?
index = torch.tensor([0, 0, 2])
src = torch.tensor([2., 3., 5.])

# %% [drill]
# Last write wins
# Predict the output of plain indexed assignment.
out = torch.zeros(3); out[index] = src
print(out)

# %% [drill]
# Buffered +=
# `out[index] += src` looks like accumulation. Predict its output.
out = torch.zeros(3); out[index] += src
print(out)

# %% [drill]
# Accumulating
# Predict the output of `index_add` and say what differs from the two above.
print(torch.zeros(3).index_add(0, index, src))

# %%
# Contract: index int64 [K] with 0 <= index < size; src [K] or [K,C], float or
# integer dtype, same device; size is a nonnegative int. Return
# [size] or [size,C] in src's dtype where slot s is the sum of src rows whose
# index is s (zero for unused slots). Do not mutate the inputs.

# %% [answer]
def scatter_sum(index: torch.Tensor, src: torch.Tensor, size: int) -> torch.Tensor:
    pass

# %%
# Baseline — The hand example, then a quick look at unused slots.
check('0114_scatter_sum_eval.py', stage='baseline')

# %%
# Follow-up — [K,C] rows, K = 0, any row order, integer exactness. Compare with
# a dense one-hot product to see what scatter_sum really computes.
check('0114_scatter_sum_eval.py', stage='followup')

# %%
# Production — Reject non-int64 or out-of-range indices, mismatched lengths,
# bad size or dtype. Gradients flow to src as grad_out[index]; inputs unchanged.
check('0114_scatter_sum_eval.py', stage='production')
