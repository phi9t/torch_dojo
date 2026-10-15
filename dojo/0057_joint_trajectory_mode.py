# %%
# 0057 · AV Motion · Select One Joint Trajectory Mode
# Adapted from Sureal joint Motion fixtures; teaching reduction, not native parity.
# One editable answer, three cumulative stages. Use Emacs/Jupyter cells.
# predictions (K,A,T,2), targets (A,T,2), valid bool (A,T).
# Return (shared mode index, scalar joint ADE); first index wins a tie.

# %%
import torch

# %%
# Baseline: mode 0 scores (0+4)/2=2; mode 1 scores (1+1)/2=1.
# Predict the shared winner; independent actor mode choice would be invalid.
predictions = torch.tensor([[[[0., 0.]], [[4., 0.]]],
                            [[[1., 0.]], [[1., 0.]]]])
targets = torch.zeros(2, 1, 2)
print('mode/actor errors:', torch.linalg.vector_norm(predictions - targets, dim=-1))

# %% [answer]
# Revise this same function through all three stages. Keep helpers here.
def joint_trajectory_mode(predictions: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor) -> tuple[int, torch.Tensor]:
    pass

# %%
print(joint_trajectory_mode(predictions, targets, torch.ones(2, 1, dtype=torch.bool)))
from torch_judge import check
check('0057_joint_trajectory_mode_eval.py', stage='baseline')

# %%
# Follow-up: valid times may have gaps, and actors have unequal valid counts.
# Average valid times within each actor, then average actors equally, then
# choose ONE mode. Ties choose the first mode. Why would pooling all actor
# time samples change the winner when observation counts differ?
probe_actor_errors = torch.tensor([[0., 100., 0.], [3., 3., 3.]])
probe_valid = torch.tensor([[True, False, True], [True, True, True]])
print('equal actor weight:', torch.stack([probe_actor_errors[a, probe_valid[a]].mean() for a in range(2)]).mean())
check('0057_joint_trajectory_mode_eval.py', stage='followup')

# %%
# Production: each actor must have >=1 valid time. An all-invalid actor,
# empty K/A/T, non-finite trajectory, dtype/device mismatch or non-bool mask
# raises ValueError. Return the first minimum and a scalar of input dtype.
check('0057_joint_trajectory_mode_eval.py', stage='production')
