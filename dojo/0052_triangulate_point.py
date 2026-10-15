# %%
# 0052 · Two-view triangulation
# Sureal-inspired geometry. Each camera maps a world column into camera
# coordinates by c = R @ X + t, with pixel homogeneous vector K @ c.

# %%
import torch

# %%
# Worked stereo probe: baseline 0.4, focal length 100, disparity 20.
# Predict Z = f*baseline/disparity = 2. At u1=10, v1=-5, the world
# point is (0.2,-0.1,2). What happens as disparity tends to zero?
print('stereo depth:', 100. * 0.4 / (10. - (-10.)))

# %%
# Baseline: two finite pixels [2], pinhole K [3,3], R [3,3], t [3];
# construct each P = K @ [R|t]. Stack u*P[2]-P[0] and v*P[2]-P[1]
# from both views. The last right singular vector is homogeneous X.

# %% [answer]
def triangulate_point(pixel1: torch.Tensor, pixel2: torch.Tensor,
                      K1: torch.Tensor, R1: torch.Tensor, t1: torch.Tensor,
                      K2: torch.Tensor, R2: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    pass

# %%
from torch_judge import check
check('0052_triangulate_point_eval.py', stage='baseline')

# %%
# Follow-up: cameras may rotate and have nonzero principal points. Keep
# the world/camera direction consistent. The two views need not be stereo.
check('0052_triangulate_point_eval.py', stage='followup')

# %%
# Production: reject shape/dtype/device errors, nonfinite values, singular
# K, near-parallel viewing rays, and a solution behind either camera with
# ValueError. Use dtype-aware tolerances; preserve dtype/device and do not
# mutate inputs. Fixed-size SVD has O(1) work per point.
check('0052_triangulate_point_eval.py', stage='production')
