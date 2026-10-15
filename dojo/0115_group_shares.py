# %%
# 0115 · Shares of the group
# Goal: give every row its share of its group's total, so shares sum to 1 per
# group. This is the weight step of a count-weighted mean (see 0108).
# Authored practice; the contracts below are authored extensions.

# %%
import torch
from torch_judge import check

# %%
# Baseline — Two voxels share a pillar: means 2 and 8, counts 3 and 1.
voxel_means = torch.tensor([2., 8.])
voxel_counts = torch.tensor([3., 1.])

# %% [drill]
# Mean of means
# Predict the plain mean of the two means, then the count-weighted mean.
print(voxel_means.mean(), (voxel_means * voxel_counts).sum() / voxel_counts.sum())

# %% [drill]
# Shares
# Predict `voxel_counts / voxel_counts.sum()`, and what `(shares * voxel_means).sum()` gives.
shares = voxel_counts / voxel_counts.sum()
print(shares, (shares * voxel_means).sum())

# %% [drill]
# Gather the total back
# Rows belong to groups [0, 1, 0]; group totals are [4, 10]. Predict `totals[group_ids]`.
group_ids = torch.tensor([0, 1, 0]); totals = torch.tensor([4., 10.])
print(totals[group_ids])

# %%
# Contract: group_ids int64 [K] in [0, num_groups); values [K], float32/64 or
# int64, all >= 0 and finite. Return float shares [K] (float64 for integer
# input): values[k] / total of its group. Every group that has rows needs a
# positive total, or raise ValueError. Empty groups are fine. Do not mutate.

# %% [answer]
def group_shares(group_ids: torch.Tensor, values: torch.Tensor, num_groups: int) -> torch.Tensor:
    pass

# %%
check('0115_group_shares_eval.py', stage='baseline')

# %%
# Follow-up — Several groups, empty groups, rows in any order. Each group's
# shares sum to 1.
check('0115_group_shares_eval.py', stage='followup')

# %%
# Production — Reject negatives, non-finite values, zero-total groups, bad ids
# and int64 totals that overflow. Three float64 values of 1e308 still give 1/3
# each: divide by something no bigger than the value BEFORE summing.
check('0115_group_shares_eval.py', stage='production')
