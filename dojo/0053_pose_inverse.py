# %%
# 0053 · Invert a rigid pose
# A 4x4 pose maps column points by p' = R @ p + t. Its inverse maps
# p' back with R.T @ (p' - t). Adapted from Sureal geometry.

# %%
import torch

# %%
# Worked probe: a quarter-turn about z sends (1,0,0) to (0,1,0).
# Translation (2,3,4) then produces (2,4,4). Why does inverse translation
# equal -R.T @ t instead of -t?
R_probe = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
t_probe = torch.tensor([2., 3., 4.])
print('forward point:', R_probe @ torch.tensor([1., 0., 0.]) + t_probe)

# %%
# Baseline: return the inverse of a finite [4,4] proper rigid pose with
# bottom row (0,0,0,1). Put R.T and -R.T@t into a new [4,4] result.

# Derive the inverse translation for the quarter-turn pose above.

# %% [answer]
def pose_inverse(pose: torch.Tensor) -> torch.Tensor:
    pass

# %%
from torch_judge import check
check('0053_pose_inverse_eval.py', stage='baseline')

# %%
# Follow-up: preserve float32/float64 dtype and device. Check both
# inverse @ pose and pose @ inverse; rotation and translation must be undone.
check('0053_pose_inverse_eval.py', stage='followup')

# %%
# Production: reject wrong shape, nonfinite entries, wrong homogeneous row,
# scaling, and reflections with ValueError. Accept small rounding error in
# proper rotations; preserve gradients and leave the input unchanged.
# Fixed 4x4 input means O(1) work and memory.
check('0053_pose_inverse_eval.py', stage='production')
