# %%
# 0051 · Camera unprojection
# Sureal-inspired geometry. R and t map world columns into camera columns:
# camera = R @ world + t. Pixels are row-stored [N,2], depths are camera z.

# %%
import torch

# %%
# Worked probe: K has fx=100, fy=50, principal point (20,10).
# At pixel (120,60) and depth 2, normalized ray is (1,1,1), so the
# identity camera returns world point (2,2,2). Predict the result with
# camera translation (1,0,0) before running your answer.
K_probe = torch.tensor([[100., 0., 20.], [0., 50., 10.], [0., 0., 1.]])
print('normalized ray:', torch.linalg.solve(K_probe, torch.tensor([120., 60., 1.])))

# %%
# Baseline: finite floating pixels [N,2], positive depth [N], invertible
# pinhole K [3,3] with last row (0,0,1). Identity R and zero t. Return
# [N,3] world points. Solve K @ ray = (u,v,1); scale the ray by depth.

# %% [answer]
def camera_unproject(pixels: torch.Tensor, depth: torch.Tensor,
                     K: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    pass

# %%
from torch_judge import check
check('0051_camera_unproject_eval.py', stage='baseline')

# %%
# Follow-up: R and t can rotate/translate the camera. For row-stored points,
# world = (camera - t) @ R. Preserve gradients through pixels and depth.
check('0051_camera_unproject_eval.py', stage='followup')

# %%
# Production: reject malformed, nonfinite, nonpositive-depth, mismatched
# dtype/device, and singular K with ValueError. Handle N=0 and noncontiguous
# pixels without mutating inputs. Output keeps input dtype/device. Work is
# O(N) for fixed 3x3 K. This reruns the earlier stages.
check('0051_camera_unproject_eval.py', stage='production')
