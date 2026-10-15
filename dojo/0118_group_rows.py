# %%
# 0118 · Group by key
# Goal: find the distinct coordinate rows of a [K,D] int64 tensor, how often
# each occurs, and which distinct row every input row belongs to.
# The grouping step of 0107 voxel pooling. Authored practice.

# %%
import torch
from torch_judge import check

# %%
# Baseline — Four rows, two columns. Think of each row as (z, y).
keys = torch.tensor([[1, 0], [0, 5], [1, 0], [0, 2]])

# %% [drill]
# Lexicographic order
# Predict the sorted distinct rows. Column 0 is the most significant.
unique, inverse, counts = torch.unique(keys, dim=0, return_inverse=True, return_counts=True)
print(unique.tolist())

# %% [drill]
# Inverse and counts
# Predict `inverse` and `counts` for the unique rows above.
print(inverse.tolist(), counts.tolist())

# %% [drill]
# Rebuild
# Predict whether `unique[inverse]` equals `keys`.
print(torch.equal(unique[inverse], keys))

# %%
# Contract: keys int64 [K,D], D >= 1, K may be 0. Return (unique [G,D] sorted
# lexicographically, inverse int64 [K] with unique[inverse] == keys, counts int64
# [G] summing to K). Do NOT pack the columns into one integer key: large
# coordinates would overflow or collide. Do not mutate the input.

# %% [answer]
def group_rows(keys: torch.Tensor):
    pass

# %%
check('0118_group_rows_eval.py', stage='baseline')

# %%
# Follow-up — Negative values, row permutations, K = 0.
check('0118_group_rows_eval.py', stage='followup')

# %%
# Production — Rows that differ only below 2**62 stay distinct. Reject float
# keys, wrong rank, D = 0. Input unchanged.
check('0118_group_rows_eval.py', stage='production')
