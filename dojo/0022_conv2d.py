# %%
# 0022 · Cross-correlation, the operation called convolution in PyTorch
# Keep one editable function. Do not call F.conv2d or nn.Conv2d in the answer.

# %%
import torch
from torch.nn import functional as F

# %%
# Baseline — Predict the result of a 2x2 all-ones kernel on this image.
image = torch.arange(9.).reshape(1, 1, 3, 3)
print("top-left window:", image[0, 0, :2, :2])
# x[B,C,H,W], weight[O,C,kH,kW] -> [B,O,H-kH+1,W-kW+1].
# This is cross-correlation: no kernel flip. Bias adds once per output channel.

# %% [answer]
def my_conv2d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None,
              stride: int = 1, padding: int = 0) -> torch.Tensor:
    pass

# %%
from torch_judge import check
check("0022_conv2d_eval.py", stage="baseline")

# %%
# Follow-up — Support non-square kernels, stride > 1, symmetric zero padding,
# and non-contiguous inputs. Match F.conv2d numerically and in gradients.
check("0022_conv2d_eval.py", stage="followup")

# %%
# Production — Reject bad ranks/channels, stride <= 0, negative padding,
# mismatched bias, and kernels larger than the padded image. Preserve dtype.
# Avoid materializing a [B,O,C,Hout,Wout,kH,kW] product tensor: unfold views
# plus contraction or bounded-window loops are both legitimate here.
check("0022_conv2d_eval.py", stage="production")
