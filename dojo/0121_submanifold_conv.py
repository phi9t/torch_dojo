# %%
# 0121 · Submanifold convolution as a rulebook
# Cheatsheet: docs/practice/submanifold-conv-cheatsheet.md (from the repository root).
# Active output rows are exactly the input rows. Neighbours that are not active
# contribute zero; do not call F.conv2d or nn.Conv2d in the answer.

# %%
from itertools import product

import torch
from torch_judge import check, get_eval_data, compare

# %%
# Baseline — Predict first.
# These active sites form a row with a hole at x=2:
#   x=0 active, x=1 active, x=2 inactive, x=3 active.
# Does a 3x3 submanifold convolution create an output row at the hole?
coords = torch.tensor([[0, 1, 0], [0, 1, 1], [0, 1, 3]])
print("active coordinates [batch, y, x]:", coords.tolist())

# absent from the output rows even though active sites on either side can see it.
eval_data = get_eval_data("0121_submanifold_conv_eval.py")
case = eval_data["dense_oracle"]
coords = case.inputs["coords"]
spatial_shape = case.inputs["spatial_shape"]
batch_size = case.inputs["batch_size"]
kernel_size = 3

# coords, spatial_shape, batch_size =

# %% [answer]
def build_rulebook(
    coords: torch.Tensor,
    spatial_shape: tuple[int, ...],
    batch_size: int,
    kernel_size: int = 3,
) -> torch.Tensor:
    """Return int64 [num_offsets, num_active_sites] source rows; -1 means absent."""
    if not isinstance(coords, torch.Tensor) or coords.dtype != torch.int64 or coords.ndim != 2:
        raise ValueError("coords must be an int64 [num_active_sites, 1 + spatial_dims] tensor")
    num_spatial_dims = coords.shape[1] - 1
    if num_spatial_dims not in (2, 3):
        raise ValueError("coords must describe 2D or 3D sites")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive Python integer")
    if type(kernel_size) is not int or kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd Python integer")
    try:
        spatial_shape = tuple(spatial_shape)
    except TypeError:
        raise ValueError("spatial_shape must contain positive Python integers") from None
    if len(spatial_shape) != num_spatial_dims or any(
        type(size) is not int or size <= 0 for size in spatial_shape
    ):
        raise ValueError("spatial_shape must match the spatial axes and contain positive integers")

    # Check using Python integers before any int64 tensor arithmetic.
    key_space = batch_size
    for size in spatial_shape:
        key_space *= size
        if key_space > torch.iinfo(torch.int64).max:
            raise ValueError("batch/spatial key space overflows int64")
    bounds = coords.new_tensor((batch_size, *spatial_shape))
    if ((coords < 0) | (coords >= bounds)).any():
        raise ValueError("coords are out of range")

    def flat_keys(coordinates: torch.Tensor) -> torch.Tensor:
        keys = coordinates[:, 0].clone()
        for axis, size in enumerate(spatial_shape, start=1):
            keys = keys * size + coordinates[:, axis]
        return keys

    # Sorted keys support batched lookup; retain original row indices for gathers.
    sorted_keys, original_rows = flat_keys(coords).sort()
    if (sorted_keys[1:] == sorted_keys[:-1]).any():
        raise ValueError("coords must be unique")
    num_active_sites = coords.shape[0]
    rulebook = torch.full(
        (kernel_size ** num_spatial_dims, num_active_sites),
        -1, dtype=torch.int64, device=coords.device,
    )
    if num_active_sites == 0:
        return rulebook

    radius = kernel_size // 2
    for offset_index, offset in enumerate(
        product(range(-radius, radius + 1), repeat=num_spatial_dims)
    ):
        valid = torch.ones(num_active_sites, dtype=torch.bool, device=coords.device)
        # Bound each axis BEFORE adding offsets or packing keys. This prevents
        # row wrapping and also avoids overflow near the int64 coordinate limit.
        for axis, (size, delta) in enumerate(zip(spatial_shape, offset), start=1):
            if delta >= 0:
                valid &= coords[:, axis] < size - delta
            else:
                valid &= coords[:, axis] >= -delta
        target_rows = valid.nonzero(as_tuple=True)[0]
        neighbors = coords[target_rows].clone()
        neighbors[:, 1:] += coords.new_tensor(offset)
        neighbor_keys = flat_keys(neighbors)
        positions = torch.searchsorted(sorted_keys, neighbor_keys)
        in_range = positions < num_active_sites
        target_rows, positions = target_rows[in_range], positions[in_range]
        matches = sorted_keys[positions] == neighbor_keys[in_range]
        rulebook[offset_index, target_rows[matches]] = original_rows[positions[matches]]
    return rulebook


def apply_rulebook(
    features: torch.Tensor,
    weight: torch.Tensor,
    rulebook: torch.Tensor,
) -> torch.Tensor:
    """Gather neighbors, mix channels and accumulate [num_active_sites, output_channels]."""
    if not all(isinstance(value, torch.Tensor) for value in (features, weight, rulebook)):
        raise ValueError("features, weight and rulebook must be tensors")
    if features.ndim != 2 or weight.ndim != 3 or rulebook.ndim != 2:
        raise ValueError("features, weight and rulebook must have ranks 2, 3 and 2")
    if features.dtype not in (torch.float32, torch.float64) or weight.dtype != features.dtype:
        raise ValueError("features and weight must share float32 or float64 dtype")
    if rulebook.dtype != torch.int64:
        raise ValueError("rulebook must be int64")
    if features.device != weight.device or features.device != rulebook.device:
        raise ValueError("features, weight and rulebook must share a device")
    num_offsets, num_active_sites = rulebook.shape
    if (num_offsets == 0 or features.shape[0] != num_active_sites
            or weight.shape[:2] != (num_offsets, features.shape[1])):
        raise ValueError("feature, weight and rulebook shapes do not match")
    if ((rulebook < -1) | (rulebook >= num_active_sites)).any():
        raise ValueError("rulebook entries must be -1 or valid feature row indices")

    output = features.new_zeros((num_active_sites, weight.shape[2]))
    for offset_index in range(num_offsets):
        source_rows = rulebook[offset_index]
        valid = source_rows >= 0
        target_rows = valid.nonzero(as_tuple=True)[0]
        # Select before gathering: -1 would otherwise select the last feature.
        contribution = features[source_rows[valid]] @ weight[offset_index]
        # Empty gathers still participate in autograd, including an empty batch.
        output = output.index_add(0, target_rows, contribution)
    return output

# %%
# Baseline — Build a 3x3 rulebook for 2D [batch, y, x] coordinates, then
# gather active neighbours and add their per-offset matrix products. The center
# offset (0, 0) maps every active output site to its own input row.
check("0121_submanifold_conv_eval.py", stage="baseline")

# %%
# Inspect the same fixtures used by the verifiers. The hole at (0, 1, 2) is
# absent from the output rows even though active sites on either side can see it.
eval_data = get_eval_data("0121_submanifold_conv_eval.py")
case = eval_data["dense_oracle"]
case.purpose, case.inputs["coords"], case.expected

# %%
# Revise the answer cell and rerun it. A correct answer returns the same
# active-row order as coords and matches this saved baseline oracle.
rulebook = build_rulebook(case.inputs["coords"], case.inputs["spatial_shape"], case.inputs["batch_size"])
actual = apply_rulebook(case.inputs["features"], case.inputs["weight"], rulebook)
result = compare(case, actual)
result.passed, result.diagnostics, result.actual

# %%
# Follow-up — Predict first.
# The first active 3D site below has an all-zero feature vector. Should the
# submanifold output still include a row for it, or does zero mean inactive?
followup = eval_data["followup_3d_batches_order"]
followup.inputs["coords"][0], followup.inputs["features"][0]

# %%
# Follow-up — The same rulebook/apply functions now infer D from coords, keep
# batches separate, preserve input row order, handle empty active sets, and keep
# gradients matching the dense oracle.
check("0121_submanifold_conv_eval.py", stage="followup")

# %%
# Production — Predict first.
# Flattening [batch, y, x] as (batch * Y + y) * X + x is only safe after every
# coordinate has been range-checked. Here X=3: the site at x=2 is not adjacent
# to the next row's x=0 site, even though adding +1 to a raw flat key lands
# there. Negative x must not wrap to the previous row either.
production_case = eval_data["production_flat_key_boundaries"]
production_case.inputs["coords"], production_case.inputs["spatial_shape"]

# %%
# Production — Reject duplicate or out-of-range coordinates and any batch/spatial
# key space that would overflow int64 before dense oracle allocation or rulebook
# lookup. Ordinary strided feature views, float32/float64 tensors, empty sparse
# inputs, and all-zero active feature rows are still valid.
check("0121_submanifold_conv_eval.py", stage="production")

# %%
# Rulebook sizing: this implementation returns int64 indices (8 bytes each).
# For 400,000 active sites and a 3x3x3 kernel, the rulebook uses 86.4 MB.
# An int32 representation would use 43.2 MB. Compare these with the feature
# payload before assuming the rulebook is the bottleneck.
num_active_sites = 400_000
num_offsets = 3 ** 3
bytes_per_index = torch.iinfo(torch.int64).bits // 8
rulebook_mb = num_offsets * num_active_sites * bytes_per_index / 1_000_000
rulebook_mb

# %%
# Exploratory extensions, not correctness gates: measure a real workload before
# making speed claims, or try a custom autograd.Function that recomputes gathers
# in backward instead of saving every per-offset slice.
