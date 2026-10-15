# %%
# 0036 · Per-output-channel INT8 linear layer
# Implement one nn.Module, then revise it as the contract grows.

# %%
import torch
import torch.nn as nn

# %%
# Baseline: finite float32 weight [out_features,in_features], both >= 1.
# Per row scale = max(abs(weight))/127; represent a zero row with scale 1.
# q = round(weight/scale).clamp(-127,127).to(int8). Store q and scale as
# buffers; dequantize in forward, then compute x @ w.T. Predict the q values
# for a row [0, 1, -1] and explain why a zero row needs special handling.
print("one row's peak magnitude:", torch.tensor([0., 1., -1.]).abs().max())

# %% [answer]
class Int8Linear(nn.Module):
    def __init__(self, weight, bias=None):
        super().__init__()
        pass

    def forward(self, x):
        pass

# %%
from torch_judge import check
check("0036_int8_quantization_eval.py", stage="baseline")

# %%
# Follow-up: optional bias is a trainable float32 Parameter [out_features].
# Support x[...,in_features], including a single vector. Each reconstructed
# weight differs from its source by at most half its own row scale (plus
# float rounding). How does per-channel scaling help when rows differ 100x?
check("0036_int8_quantization_eval.py", stage="followup")

# %%
# Production: reject malformed weight, bias, and input shape/dtype/device;
# reject NaN/inf. Constructor must not mutate weight or bias. Buffers survive
# state_dict round trips and device moves. Gradients flow to x and bias, not
# quantized weights. This reference dequantizes for matmul; it teaches
# storage quantization, not an optimized integer GEMM.
check("0036_int8_quantization_eval.py", stage="production")
