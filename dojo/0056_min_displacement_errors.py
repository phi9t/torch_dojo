# %%
# 0056 · AV Motion · Minimum Displacement Errors
# Adapted from Sureal motion fixtures. This teaching metric uses an explicit autonomy evaluation contract.
# One editable answer, three cumulative stages; run cells in Emacs/Jupyter.
# predictions (K,T,2), target (T,2), valid bool (T,).
# Return scalar (minADE, minFDE); the two minima may choose different modes.

# %%
import torch

# %%
# Baseline: mode 0 has distances [0,0,3]; mode 1 has [2,2,0].
# Predict minADE=1 and minFDE=0 before implementing.
predictions = torch.tensor([[[0., 0.], [0., 0.], [3., 0.]],
                            [[2., 0.], [2., 0.], [0., 0.]]])
target = torch.zeros(3, 2)
print('distances:', torch.linalg.vector_norm(predictions - target, dim=-1))

# %% [answer]
# Revise this same function through all three stages. Keep helpers here.
def min_displacement_errors(predictions: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pass

# %%
print(min_displacement_errors(predictions, target, torch.ones(3, dtype=torch.bool)))
from torch_judge import check
check('0056_min_displacement_errors_eval.py', stage='baseline')

# %%
# Follow-up: valid can be non-prefix, e.g. [True,False,True,False].
# Omit invalid times from ADE; FDE uses the last valid index, not T-1.
# For distances [1,100,2,100], predict ADE=1.5 and FDE=2.
probe_errors = torch.tensor([1., 100., 2., 100.])
probe_valid = torch.tensor([True, False, True, False])
print('masked mean and last valid:', probe_errors[probe_valid].mean(), probe_errors[probe_valid][-1])
check('0056_min_displacement_errors_eval.py', stage='followup')

# %%
# Production: require K>=1, T>=1, finite matching floating trajectories,
# bool mask on same device, at least one valid step; invalid input raises
# ValueError. Keep inputs unchanged and return scalar tensors in input dtype.
check('0056_min_displacement_errors_eval.py', stage='production')
