# %%
# 0054 · LiDAR polar coordinates
# Range r, azimuth a, elevation e map to (r*cos(e)*cos(a),
# r*cos(e)*sin(a), r*sin(e)). Angles are radians; azimuth turns from +x to
# +y, elevation rises from the xy plane. Adapted from Sureal geometry.

# %%
import torch

# %%
# Worked probe: predict the +x, +y, +z endpoints before running this cell.
axis_rays = torch.tensor([[2., 0., 0.], [2., torch.pi / 2, 0.],
                          [2., 0., torch.pi / 2]])
print('axis rays:', axis_rays)

# %%
# Baseline: convert finite [N,3] polar rows to [N,3] Cartesian rows;
# zero range maps to origin. Split horizontal r*cos(e) between x and y.

# %% [answer]
def polar_to_cartesian(polar: torch.Tensor) -> torch.Tensor:
    pass

# %%
from torch_judge import check
check('0054_polar_to_cartesian_eval.py', stage='baseline')

# %%
# Follow-up: arbitrary leading dimensions (...,3), including an empty
# batch, and negative elevation. Preserve shape, dtype, and gradients.
check('0054_polar_to_cartesian_eval.py', stage='followup')

# %%
# Production: reject negative range, nonfinite values, malformed shape,
# and unsupported dtypes with ValueError. Support noncontiguous input
# without mutation. Vectorized O(number of returns) work/output memory.
check('0054_polar_to_cartesian_eval.py', stage='production')
