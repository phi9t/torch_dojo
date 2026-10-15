# %%
# 0044 · Pinhole camera projection
# For row-stored world points, camera = points @ R.T + t. Project with K,
# then divide homogeneous x/y by homogeneous z. Return (uv[N,2], depth[N]);
# depth is camera z. K has last row [0,0,1].

# %%
import torch

# %%
# Worked probe — predict the output before running this cell.
p = torch.tensor([[2., 4., 2.]])
print("normalized coordinates:", p[:, :2] / p[:, 2:])
# Predict: [1,2]. What changes if camera translation is [1,0,0]?

# %%
# Baseline contract
# Project finite floating [N,3] points through a pinhole camera with
# identity R/t or simple intrinsics. Return camera-space depth separately.
# Work through shapes and one boundary case before editing the answer.

# %% [answer]
def camera_project(points: torch.Tensor, K: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pass

# %%
from torch_judge import check
check('0044_camera_project_eval.py', stage='baseline')

# %%
# Follow-up contract
# Add nonidentity R, t, and K. Keep row versus column-vector order straight;
# make the output differentiable with respect to points.
# Revise the same answer cell, then run this cumulative check.
check('0044_camera_project_eval.py', stage='followup')

# %%
# Production contract
# Reject nonpositive depth and malformed/mismatched tensors with ValueError.
# Handle N=0, noncontiguous input, and no mutation. O(N) time/output space.
# No image bounds are available here, so do not invent a visibility mask.
# This check reruns baseline and follow-up before production verifiers.
check('0044_camera_project_eval.py', stage='production')
