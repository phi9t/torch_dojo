# %%
# 0127 · Conv2D from scratch in NumPy (practice set 3/7, 20 min)
# Cheatsheet: docs/practice/conv2d-cheatsheet.md (from the repository root).
# Goal: implement `conv2d` as cross-correlation. No convolution helper. Then
# explain a patch-extraction (im2col) + matrix-contraction version and its
# memory cost: one row per output location and one column per input-channel
# and kernel-position combination. With matching input/output spatial sizes,
# it holds kernel_height * kernel_width times the input elements.

# %%
import numpy as np

# %%
# Baseline — X: [batch_size,input_channels,input_height,input_width].
# Filters W: [output_channels,input_channels,kernel_height,kernel_width].
# Bias: [output_channels].
# Symmetric zero padding, positive strides, dilation=1, groups=1.
# Derive: output_height = floor((input_height + 2*padding_height - kernel_height) / stride_height) + 1, output_width likewise.
# Cross-correlation: do NOT flip the filter.
# Hand-checkable: [[1,2,3],[4,5,6],[7,8,9]] with filter [[1,0],[0,-1]] -> 2x2 of -4.

# %% [drill]
# Output size
# input_height=8, input_width=9, kernel (3,4), stride (2,3), padding (1,2). Predict (output_height, output_width).
print((8 + 2 - 3) // 2 + 1, (9 + 4 - 4) // 3 + 1)

# %% [drill]
# Flipped or not?
# Predict the single output of a 2x2 input [[0,1],[2,3]] with filter [[1,2],[3,4]].
print(0 * 1 + 1 * 2 + 2 * 3 + 3 * 4, "vs flipped:", 0 * 4 + 1 * 3 + 2 * 2 + 3 * 1)

# %% [drill]
# Memory of im2col
# batch_size=8, input_channels=64, 56x56 input, 3x3 kernel, stride 1, pad 1, float64: how many
# times larger is the patch matrix than the input?
print(3 * 3)

# %%
# Inputs conforming to the answer's signature.
rng = np.random.default_rng(0)
X: np.ndarray = rng.normal(size=(2, 3, 7, 6))     # [batch_size,input_channels,H,W]
W: np.ndarray = rng.normal(size=(4, 3, 3, 3))     # [output_channels,input_channels,kernel_height,kernel_width]
bias: np.ndarray = rng.normal(size=4)             # [output_channels]
stride: tuple[int, int] = (2, 1)
padding: tuple[int, int] = (1, 1)

# %% [answer]
def conv2d(X: np.ndarray, W: np.ndarray, bias: np.ndarray,
           stride: tuple[int, int] = (1, 1), padding: tuple[int, int] = (0, 0)) -> np.ndarray:
    # X: [batch_size,input_channels,input_height,input_width]
    # W: [output_channels,input_channels,kernel_height,kernel_width]
    # bias: [output_channels]
    # Returns float64 [batch_size,output_channels,output_height,output_width].
    X, W, bias = (np.asarray(a, dtype=np.float64) for a in (X, W, bias))
    if X.ndim != 4 or W.ndim != 4:
        raise ValueError("X and W must be four-dimensional")
    batch_size, input_channels, input_height, input_width = X.shape
    output_channels, filter_channels, kernel_height, kernel_width = W.shape
    if filter_channels != input_channels or bias.shape != (output_channels,):
        raise ValueError("filters must match input channels; bias must be [output_channels]")
    if kernel_height < 1 or kernel_width < 1:
        raise ValueError("kernel dimensions must be positive")

    def spatial_pair(value, name, minimum):
        try:
            pair = tuple(value)
        except TypeError:
            raise ValueError(f"{name} must contain two integers") from None
        if len(pair) != 2 or any(
            isinstance(v, (bool, np.bool_))
            or not isinstance(v, (int, np.integer))
            or v < minimum
            for v in pair
        ):
            raise ValueError(f"{name} must contain two integers >= {minimum}")
        return tuple(int(v) for v in pair)

    stride_height, stride_width = spatial_pair(stride, "stride", 1)
    padding_height, padding_width = spatial_pair(padding, "padding", 0)
    output_height = (input_height + 2 * padding_height - kernel_height) // stride_height + 1
    output_width = (input_width + 2 * padding_width - kernel_width) // stride_width + 1
    if output_height < 1 or output_width < 1:
        raise ValueError("kernel larger than padded input")

    # Pad height and width with zeros; leave batch and channels alone.
    Xp = np.pad(
        X,
        ((0, 0), (0, 0), (padding_height, padding_height), (padding_width, padding_width)),
    )
    out = np.zeros((batch_size, output_channels, output_height, output_width), dtype=np.float64)
    out += bias[None, :, None, None]
    # Loop only over kernel taps. Each slice covers every output position;
    # einsum sums input channels while retaining batch and output channels.
    # This avoids materializing the full im2col matrix.
    for i in range(kernel_height):
        for j in range(kernel_width):
            patch = Xp[
                :, :,
                i:i + stride_height * output_height:stride_height,
                j:j + stride_width * output_width:stride_width,
            ]  # [batch_size,input_channels,output_height,output_width]
            out += np.einsum("bchw,oc->bohw", patch, W[:, :, i, j])
    return out

# %%
from torch_judge import check
check("0127_conv2d_scratch_eval.py", stage="baseline")

# %%
# Follow-up — Multiple input/output channels, stride 2, nonzero padding, and
# non-square kernels all agree with a slow six-loop reference.
check("0127_conv2d_scratch_eval.py", stage="followup")

# %%
# Production — Do not mutate inputs; padding must be zeros, not edge values;
# reject channel mismatch, bad bias shape, non-positive stride, negative
# padding and kernels larger than the padded input with ValueError.
# Compute batches/channels/output positions together; keep only kernel-tap loops.
# Stride/padding pairs must be integers, not floats or booleans. Non-contiguous
# inputs are supported, with float64 output and no caller-input mutation.
check("0127_conv2d_scratch_eval.py", stage="production")
