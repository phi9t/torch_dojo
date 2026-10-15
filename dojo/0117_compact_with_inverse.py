# %%
# 0117 · Compact and put back
# Goal: keep only the real rows of a padded [B,N,C] tensor, remember where each
# came from, and give every original slot its compact row id (or -1).
# The mask -> nonzero -> inverse step in 0107. Authored practice.

# %%
import torch
from torch_judge import check

# %%
# Baseline — B=2, N=3, with some padding slots.
x = torch.arange(6.).reshape(2, 3, 1)
mask = torch.tensor([[True, False, True], [False, True, True]])

# %% [drill]
# Addresses
# Predict `mask.nonzero()`: which (b, n) pairs, and in what order?
print(mask.nonzero().tolist())

# %% [drill]
# Same order?
# Predict `x[mask][:, 0]`. Does it line up with the addresses above?
print(x[mask][:, 0].tolist())

# %% [drill]
# The inverse
# Predict a [2,3] tensor holding each kept slot's compact row id and -1 elsewhere.

# %%
# Contract: x float [B,N,C], mask bool [B,N]. Return (compact [M,C],
# addr int64 [M,2] of (b, n), inverse int64 [B,N]) with M = mask.sum(). Rows are
# in row-major (batch, then slot) order. inverse[b,n] is the compact row of a
# kept slot, else -1. Masked-out contents (even NaN) never reach the outputs.
# Gradients flow from compact to x. Do not mutate the inputs.

# %% [answer]
def compact_with_inverse(x: torch.Tensor, mask: torch.Tensor):
    pass

# %%
check('0117_compact_with_inverse_eval.py', stage='baseline')

# %%
# Follow-up — Scatter back to rebuild x, all-False masks, NaN padding.
check('0117_compact_with_inverse_eval.py', stage='followup')

# %%
# Production — Reject non-bool masks and shape mismatches; gradients reach
# exactly the kept rows; non-contiguous x works; inputs unchanged.
check('0117_compact_with_inverse_eval.py', stage='production')
