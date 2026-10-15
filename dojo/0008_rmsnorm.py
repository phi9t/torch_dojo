# %%
# 0008 · RMSNorm from a vector to mixed precision

# %%
import torch

# %%
# Worked probe: [3,4] has mean square 12.5. RMSNorm does NOT subtract
# the mean, so a positive constant vector remains positive.
probe = torch.tensor([[3., 4.]])
print("mean square:", probe.square().mean(-1, keepdim=True))

# Predict the normalized [3,4] row when weight=[1,1] and eps=0.

# %% [answer]
def rms_norm(x, weight, eps=1e-6):
    # x [...,D], weight [D]; replace this body. No built-in norm layer.
    pass

# %%
# Baseline: normalize each row on the last axis with
# x / sqrt(mean(x**2)+eps) * weight. eps is inside the square root.
from torch_judge import check
check("0008_rmsnorm_eval.py", stage="baseline")

# %%
# Follow-up: accept arbitrary leading axes and noncontiguous views;
# preserve shape and provide finite gradients for x and weight.
# A zero row stays zero; a constant row is not centered.
check("0008_rmsnorm_eval.py", stage="followup")

# %%
# Production: reject invalid feature width/weight shape and eps<=0.
# For float16/bfloat16, compute the mean square in float32 and cast
# normalized output back to x's dtype. This protects large fp16 values
# from squaring overflow while keeping the public output dtype.
check("0008_rmsnorm_eval.py", stage="production")
