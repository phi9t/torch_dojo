# %%
# 0045 · Rigid point alignment
# Find proper R,t minimizing paired squared errors under target ≈ source @ R.T+t.
# Use centered covariance and SVD. A reflection is not a valid rotation.

# %%
import torch

# %%
# Worked probe — predict the output before running this cell.
source = torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
target = source + torch.tensor([2.,-1.,3.])
print("centroid shift:", target.mean(0) - source.mean(0))
# Predict: identity R and t=[2,-1,3].

# %%
# Baseline contract
# Use matching floating [N,3] pairs, N>=3. Return R[3,3], t[3],
# with source @ R.T+t close to target and det(R)=+1.
# Work through shapes and one boundary case before editing the answer.

# %% [answer]
def rigid_align(source: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pass

# %%
from torch_judge import check
check('0045_rigid_align_eval.py', stage='baseline')

# %%
# Follow-up contract
# Correct a reflected target, and reject N<3. Which SVD factor gets its
# last sign flipped when the unconstrained transform has det<0?
# Revise the same answer cell, then run this cumulative check.
check('0045_rigid_align_eval.py', stage='followup')

# %%
# Production contract
# Reject nonfinite, mismatched, or collinear configurations: rotation is
# under-determined for collinear points. Fit noisy pairs in least squares,
# keep inputs unchanged, and explain the O(N) covariance + constant 3x3 SVD.
# This check reruns baseline and follow-up before production verifiers.
check('0045_rigid_align_eval.py', stage='production')
