# %%
# 0027 · ViT patch embedding
# Keep one editable class and grow it through three stages.

# %%
import torch
from torch import nn
from torch_judge import check, get_eval_data, compare, replay

# %%
# Baseline — A 4x4 single-channel image with 2x2 patches yields four
# row-major flattened patch vectors. Predict them before running.
image = torch.arange(16.).reshape(1, 1, 4, 4)
print("top-left patch:", image[0, 0, :2, :2].flatten())
# Build nn.Linear(C*P*P, E) and num_patches=(img_size//P)**2.
# Return [B,N,E], with channels inside each row-major patch.

# %% [answer]
class PatchEmbedding(nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

# %%
check("0027_vit_patch_eval.py", stage="baseline")

# %%
# Follow-up — Test a numbered multichannel image; patch order and channel
# order matter. Non-contiguous input should work. Gradients must reach input
# and projection. Support non-square runtime H/W when both divide by P.
check("0027_vit_patch_eval.py", stage="followup")

# %%
# Production — Validate constructor dimensions and reject wrong channels or
# H/W not divisible by P instead of silently dropping border pixels. Preserve
# dtype/device with .to(...). Keep [B,0,E] valid if a zero batch is supplied.
check("0027_vit_patch_eval.py", stage="production")

# %%
# Inspectable fixtures carry constructor arguments and explicit shared weights.
# Randomly initialized projections cannot be compared to these fixed expected
# values until you copy the supplied weights. The model remains your class.
eval_data = get_eval_data("0027_vit_patch_eval.py")
case = eval_data["numbered_channels"]
case.inputs, case.generation, case.expected

# %%
model = PatchEmbedding(*case.generation["constructor"]).double()
with torch.no_grad():
    model.proj.weight.copy_(case.generation["weight"])
    model.proj.bias.copy_(case.generation["bias"])
actual = model(**case.inputs)
first_result = compare(case,actual)
first_result.passed, first_result.diagnostics, first_result.actual, first_result.expected

# %%
# Edit/rerun your class cell, construct a NEW model to pick up that definition,
# then copy the SAME weights before fresh replay. Retain first_result to inspect
# the earlier attempt. A new class definition does not upgrade an old instance.
model = PatchEmbedding(*case.generation["constructor"]).double()
with torch.no_grad():
    model.proj.weight.copy_(case.generation["weight"])
    model.proj.bias.copy_(case.generation["bias"])
results = replay("0027_vit_patch_eval.py",model)
failures = {name:r for name,r in results.items() if not r.passed}
failures

# %%
# In the eval script, run PatchEmbeddingLoop, PatchEmbeddingUnfold, and
# PatchEmbeddingConv as complete versions. They expose the SAME .proj Linear
# weights, so copy state_dict before comparing. Convolution reshapes those
# weights into [E,C,P,P] and uses stride=P without padding; it is cross-correlation.
# benchmark_patch_embedding() compares shared weights and reports numerical error,
# workload/timing variability and analytical patch-matrix storage. Backend
# workspace and retained backward storage are separate; no speed guarantee.
# .num_patches is the constructor's nominal square count. Runtime output uses
# actual H/P and W/P, which can be rectangular. Empty batch is [0,N,E].
# Connect tokens to 0110 with grid_hw=(H//P,W//P), not sqrt(token_count).
# For THIS unpadded nonoverlapping extractor, patch-center image coordinates
# are (P*column+(P-1)/2, P*row+(P-1)/2). Its feature_from_image affine scales
# by 1/P and translates by -(P-1)/(2P). Other backbones need explicit metadata;
# tensor sizes alone do not establish receptive-field or preprocessing alignment.
