# %%
# 0042 · Pixel-center resizing
# Map pixel centers through resize and padding with align_corners=False.
# Coordinates are (...,2) floating (x,y); sizes are positive integer (height,width),
# padding is integer (x,y). Keep one answer and extend it at each check.

# %%
import torch

# %%
# Worked probe — predict the output before running this cell.
p = torch.tensor([[0., 0.], [3., 1.]])
print("resize probe:", (p + .5) * torch.tensor([2., 2.]) - .5 + torch.tensor([1., 2.]))
# Predict: [[1.5, 2.5], [7.5, 4.5]]. Why add .5 before scaling?

# %%
# Baseline contract
# Forward-map x with width scale and y with height scale; add padding last.
# Return the same coordinate shape, dtype, and device.
# Work through shapes and one boundary case before editing the answer.

# %% [answer]
def pixel_centers(points: torch.Tensor, original_hw: tuple[int, int], resized_hw: tuple[int, int], pad_xy: tuple[int, int], inverse: bool = False) -> torch.Tensor:
    pass

# %%
from torch_judge import check
check('0042_pixel_centers_eval.py', stage='baseline')

# %%
# Follow-up contract
# Add inverse=True by undoing padding before undoing the half-pixel scale.
# Handle anisotropic scales, batched (...,2) points, and an empty leading dimension.
# Revise the same answer cell, then run this cumulative check.
check('0042_pixel_centers_eval.py', stage='followup')

# %%
# Production contract
# Preserve gradients and inputs. Reject malformed coordinates, nonfinite values,
# and zero/negative image sizes with ValueError. O(P) time and output space;
# no per-pixel grid allocation.
# This check reruns baseline and follow-up before production verifiers.
check('0042_pixel_centers_eval.py', stage='production')
