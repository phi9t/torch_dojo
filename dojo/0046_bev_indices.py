# %%
# 0046 · BEV grid indexing
# Map floating xy[N,2] to integer grid cells. Grid bounds are half-open:
# minimum inclusive, maximum exclusive. Invalid points get (-1,-1) and False.

# %%
import torch

# %%
# Worked probe — predict the output before running this cell.
xy = torch.tensor([[-1.,-1.],[0.,0.],[1.,1.]])
print("candidate offsets:", (xy + 1.) / .5)
# Predict: first cell 0, middle cell 2, maximum invalid.

# %%
# Baseline contract
# Use floor((xy-minimum)/resolution). Check half-open validity before
# flooring; return int64 indices [N,2] and bool validity [N].
# Work through shapes and one boundary case before editing the answer.

# %% [answer]
def bev_indices(points_xy: torch.Tensor, minimum: torch.Tensor, maximum: torch.Tensor, resolution: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pass

# %%
from torch_judge import check
check('0046_bev_indices_eval.py', stage='baseline')

# %%
# Follow-up contract
# Add anisotropic resolution and a partial last cell. Interior points
# that round to the computed upper cell boundary must clamp to the last cell.
# Revise the same answer cell, then run this cumulative check.
check('0046_bev_indices_eval.py', stage='followup')

# %%
# Production contract
# Reject malformed/nonfinite geometry and nonpositive resolution with
# ValueError. Handle empty/noncontiguous inputs without mutation. O(N) time
# and memory; do not build a dense occupancy grid.
# This check reruns baseline and follow-up before production verifiers.
check('0046_bev_indices_eval.py', stage='production')
