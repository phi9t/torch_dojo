# %%
# 0058 · AV Planning · First Sampled Collision
# Inspired by Sureal overlap checks; uses circular sampled footprints, not native parity.
# One editable answer, three cumulative stages. Work in Emacs/Jupyter.
# ego (T,2), others (A,T,2), other_radii (A,). Return earliest step or -1.

# %%
import torch

# %%
# Baseline: circular footprints contact when center distance <= sum radii.
# The second actor contacts at t=1. Predict the earliest step.
ego = torch.tensor([[0., 0.], [1., 0.], [2., 0.]])
others = torch.tensor([[[5., 0.], [4., 0.], [2.9, 0.]],
                       [[5., 0.], [1.9, 0.], [5., 0.]]])
print('distances:', torch.linalg.vector_norm(others - ego[None], dim=-1))

# %% [answer]
# Revise this same function through all three stages. Keep helpers here.
def first_collision_step(ego: torch.Tensor, others: torch.Tensor, ego_radius: float, other_radii: torch.Tensor) -> int:
    pass

# %%
print(first_collision_step(ego, others, .5, torch.tensor([.5, .5])))
from torch_judge import check
check('0058_first_collision_step_eval.py', stage='baseline')

# %%
# Follow-up: paths crossing at different sampled times do not collide.
# Each actor uses its own radius; exact tangency DOES collide.
# There is no interpolation between sampled positions.
probe_ego = torch.tensor([[0., 0.], [1., 0.]])
probe_other = torch.tensor([[1., 0.], [0., 0.]])
print('different-time crossing distances:', torch.linalg.vector_norm(probe_other - probe_ego, dim=-1))
check('0058_first_collision_step_eval.py', stage='followup')

# %%
# Production: no actors or no time steps returns -1. Reject negative/non-finite
# radii, non-finite coordinates, shape and floating dtype/device mismatches
# with ValueError. Do not alter any input.
check('0058_first_collision_step_eval.py', stage='production')
