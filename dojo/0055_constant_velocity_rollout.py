# %%
# 0055 · AV Motion · Constant-Velocity Rollout
# Adapted from Sureal: experiments/3d-pathway/pipeline/dynamic.py
# One editable answer, three cumulative stages. Work top to bottom in Jupyter.
# Signature: constant_velocity_rollout(history, times, future_times) -> (H, 2)
# All tensors are on the same device with the same floating dtype.

# %%
import torch

# %%
# Baseline: two observations and one future query. Velocity is distance / seconds.
# Work it out: 2 metres in 0.5 seconds gives 4 metres/second.
# From (2,-1) at t=1.5, where is the actor at t=2?
history = torch.tensor([[0., 0.], [2., -1.]])
times = torch.tensor([1., 1.5])
print('displacement and elapsed time:', history[-1] - history[-2], times[-1] - times[-2])

# %% [answer]
# Revise this same function through all three stages. Keep any helper definitions here.
def constant_velocity_rollout(history: torch.Tensor, times: torch.Tensor, future_times: torch.Tensor) -> torch.Tensor:
    pass  # return a tensor of shape (H, 2)

# %%
# Predict (4,-2), then try your answer and the baseline check.
print(constant_velocity_rollout(history, times, torch.tensor([2.])))
from torch_judge import check
check('0055_constant_velocity_rollout_eval.py', stage='baseline')

# %%
# Follow-up: history and queries can be irregularly spaced. Use only the last
# two observations for velocity; reject duplicate, reversed, or past timestamps.
# Predict both points before running this cell.
history = torch.tensor([[99., 99.], [1., 2.], [3., 2.]])
times = torch.tensor([0., 1., 3.])
print(constant_velocity_rollout(history, times, torch.tensor([3.5, 5.])))
check('0055_constant_velocity_rollout_eval.py', stage='followup')

# %%
# Production: require finite positions/times, >=2 observations, >=1 future
# query, strict time order, matching floating dtype/device, and no mutation.
# A non-finite query or a mismatched time dtype raises ValueError.
check('0055_constant_velocity_rollout_eval.py', stage='production')
