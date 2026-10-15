# %%
# 0107 · Sparse voxel/pillar pooling: collisions are contributions
# Baseline 30–45 minutes. Keep revising the single answer cell.
# This adds Torch gradients; the earlier NumPy 0066 API remains unchanged.

# %%
import torch
from torch_judge import check, get_eval_data, compare, replay

# %%
# Step 1 — Work one occupied voxel by hand.
# Features [2,0] and [4,0] share a cell; their count is 2 and mean is [3,0].
# A feature [0,0] still occupies its voxel. Missing is represented by inverse
# -1, not by inspecting whether the feature happens to equal zero.
# Does assigning out[cell] = feature preserve repeated points? Explain why.

# %%
# Step 2 — Contract:
# points [B,N,3], features [B,N,C], point_valid bool [B,N]; B,C positive, N>=0.
# origin/voxel_size [3] xyz; grid_shape int64 [3] xyz, all same device.
# Common float32/float64 floating dtype; finite valid points/features, finite
# origin, positive finite size, positive shape; finite distinct physical bounds.
# Include origin <= point < origin+size*shape (HALF OPEN), then floor cell xyz.
# Outputs: lexicographically sorted int64 indices [V,4] in BATCH,Z,Y,X order,
# int64 counts [V], means [V,C] in input dtype, inverse int64 [B,N].
# Inverse contains GLOBAL occupied-row IDs, or -1 for excluded points.
# Wrong metadata/shapes/types raise ValueError. No dense grid or packed key.

# %% [answer]
def pool_points(points, features, point_valid, origin, voxel_size, grid_shape,
                pillar=False):
    # Dims: b=batch, n=input point slot, c=channel, 3=xyz, 2=[batch,n], 4=[batch,z,y,x],
    #       m=included point (valid and inside the grid), FLATTENED across batches:
    #         m = sum(included_bn), and the batch id lives in column 0 of *_m2 / *_m4,
    #       v=occupied voxel/pillar row, also across batches (batch is column 0 of voxel_inds_v4)
    # Suffix order matches shape; "_inds_" holds integer indices, "_row_inds_" row ids in v.
    point_xyz_bn3, point_features_bnc, point_valid_bn = points, features, point_valid
    tensors = (point_xyz_bn3, point_features_bnc, point_valid_bn, origin, voxel_size, grid_shape)

    # Validation: all ValueError, before any arithmetic on the data
    if not all(isinstance(t, torch.Tensor) for t in tensors) or not isinstance(pillar, bool):
        raise ValueError("tensor inputs and bool pillar required")
    if point_xyz_bn3.ndim != 3 or point_xyz_bn3.shape[-1] != 3 or point_xyz_bn3.shape[0] < 1:
        raise ValueError("points must be [B,N,3] with B>0")
    batch_size, num_points, _ = point_xyz_bn3.shape
    if (point_features_bnc.ndim != 3 or point_features_bnc.shape[:2] != (batch_size, num_points)
            or point_features_bnc.shape[2] < 1 or point_valid_bn.shape != (batch_size, num_points)):
        raise ValueError("features/point_valid shape mismatch")
    if any(t.shape != (3,) for t in (origin, voxel_size, grid_shape)):
        raise ValueError("origin, voxel_size, grid_shape must be [3]")
    dtype = point_xyz_bn3.dtype
    if dtype not in (torch.float32, torch.float64) or any(t.dtype != dtype for t in (point_features_bnc, origin, voxel_size)):
        raise ValueError("floating tensors must share one float32/float64 dtype")
    if point_valid_bn.dtype != torch.bool or grid_shape.dtype != torch.int64:
        raise ValueError("point_valid must be bool and grid_shape int64")
    if any(t.device != point_xyz_bn3.device for t in tensors):
        raise ValueError("tensors must share a device")
    if not (torch.isfinite(origin).all() and torch.isfinite(voxel_size).all()
            and (voxel_size > 0).all() and (grid_shape > 0).all()):
        raise ValueError("finite origin and positive voxel_size/grid_shape required")
    upper_bound_xyz_3 = origin + voxel_size * grid_shape
    if not torch.isfinite(upper_bound_xyz_3).all() or not (upper_bound_xyz_3 > origin).all():
        raise ValueError("bounds must be finite and distinct in the working dtype")
    if not (torch.isfinite(point_xyz_bn3[point_valid_bn]).all()
            and torch.isfinite(point_features_bnc[point_valid_bn]).all()):
        raise ValueError("valid points and features must be finite")

    # Membership is decided in PHYSICAL coordinates (half-open) before any rounding.
    # Padding rows are replaced by origin first so NaNs never reach arithmetic.
    filled_xyz_bn3 = torch.where(point_valid_bn[..., None], point_xyz_bn3, origin)
    included_bn = (point_valid_bn & (filled_xyz_bn3 >= origin).all(-1)
                   & (filled_xyz_bn3 < upper_bound_xyz_3).all(-1))
    # Flatten (b, n) -> m: one row per included point, across ALL batches, in row-major
    # (batch-major, then point) order. The batch axis is not kept as a tensor axis; its
    # id moves into column 0 of the [m,2] result, so m = included_bn.sum() (e.g. 10 for
    # B=2, N=5 with every point included) and column 1 is the point slot n.
    included_batch_point_inds_m2 = included_bn.nonzero(as_tuple=False)
    # Boolean-mask indexing flattens (b, n) the same way, so rows line up with the m2 above:
    included_xyz_m3 = filled_xyz_bn3[included_bn]                # [b,n,3] -> [m,3]
    included_features_mc = point_features_bnc[included_bn]       # [b,n,c] -> [m,c]

    cell_xyz_inds_m3 = torch.floor((included_xyz_m3 - origin) / voxel_size)
    if not (torch.isfinite(cell_xyz_inds_m3).all() and (cell_xyz_inds_m3 >= 0).all() and (cell_xyz_inds_m3 < 2**63).all()):
        raise ValueError("cell coordinate not representable as int64")
    # Rounding may push an interior point onto the upper edge: keep it in the last cell
    cell_xyz_inds_m3 = torch.minimum(cell_xyz_inds_m3.to(torch.int64), grid_shape - 1)
    cell_z_inds_m = torch.zeros_like(cell_xyz_inds_m3[:, 2]) if pillar else cell_xyz_inds_m3[:, 2]
    group_keys_m4 = torch.stack(
        (included_batch_point_inds_m2[:, 0], cell_z_inds_m, cell_xyz_inds_m3[:, 1], cell_xyz_inds_m3[:, 0]), dim=1)

    # Group by row: sorted lexicographically in (batch, z, y, x); no packed key
    voxel_inds_v4, voxel_row_inds_m, voxel_counts_v = torch.unique(
        group_keys_m4, dim=0, sorted=True, return_inverse=True, return_counts=True)
    inverse_bn = torch.full((batch_size, num_points), -1, dtype=torch.int64, device=point_xyz_bn3.device)
    inverse_bn[included_batch_point_inds_m2[:, 0], included_batch_point_inds_m2[:, 1]] = voxel_row_inds_m

    # Mean without overflow: widen float32, scale each channel by its group max
    # magnitude, average the bounded values, clamp to [-1,1], restore the scale.
    work_features_mc = included_features_mc.double() if dtype == torch.float32 else included_features_mc
    num_voxels, num_channels = voxel_inds_v4.shape[0], work_features_mc.shape[1]
    max_abs_vc = work_features_mc.new_zeros(num_voxels, num_channels).scatter_reduce(
        0, voxel_row_inds_m[:, None].expand_as(work_features_mc), work_features_mc.abs(), reduce="amax", include_self=True)
    safe_max_abs_vc = torch.where(max_abs_vc > 0, max_abs_vc, torch.ones_like(max_abs_vc))
    bounded_sum_vc = work_features_mc.new_zeros(num_voxels, num_channels).index_add(
        0, voxel_row_inds_m, work_features_mc / safe_max_abs_vc[voxel_row_inds_m])
    voxel_means_vc = ((bounded_sum_vc / voxel_counts_v[:, None]).clamp(-1, 1) * safe_max_abs_vc).to(dtype)
    return voxel_inds_v4, voxel_counts_v, voxel_means_vc, inverse_bn


# %%
check('0107_voxel_pillar_pool_eval.py', stage='baseline')

# %%
# Inspect the actual fixture and keep the mismatch record around as you revise.
eval_data = get_eval_data('0107_voxel_pillar_pool_eval.py')
case = eval_data['two_voxels_and_boundaries']
case.inputs, case.expected

# %%
actual = pool_points(**case.inputs)
result = compare(case, actual)
result.passed, result.diagnostics, result.actual, result.expected

# %%
# Edit the answer and rerun it. Fresh replay recreates inputs; result above
# remains evidence of the older implementation.
results = replay('0107_voxel_pillar_pool_eval.py', pool_points,
                 names=['two_voxels_and_boundaries', 'pillar_collapses_z'])
failures = {name:r for name,r in results.items() if not r.passed}
failures

# %%
# Step 3 — Pillars collapse GROUPING z to 0, but still reject points outside
# the declared 3D grid. Never merge batches. Predict the count-weighted result
# when the z=0 voxel has two points and the z=1 voxel has one.
# Permuting input points preserves indices/counts/means, while inverse permutes
# with the points. Duplicate points contribute multiple times.
check('0107_voxel_pillar_pool_eval.py', stage='followup')

# %%
# Step 4 — Empty points, all excluded, strided features and masked NaNs.
# Select/sanitize padding before arithmetic so it cannot poison gradients.
# Mean gradients give each included point 1/count per channel. Coordinates,
# floor membership and integer row IDs are discrete; no coordinate derivative
# promised through grouping changes. Preserve inputs.
# Widen float32 and scale by per-group/channel max magnitude before summing.
# Clamp the normalized mean to [-1,1] so rounding cannot overflow finfo.max.
# Boundary correction gradients follow clamp/max; smooth interior means retain
# 1/count derivatives. Retain occupied zeros and counts.
check('0107_voxel_pillar_pool_eval.py', stage='production')

# %%
# Run backward on LIVE outputs, not detached comparison snapshots.
case = get_eval_data('0107_voxel_pillar_pool_eval.py')['two_voxels_and_boundaries']
case.inputs['features'].requires_grad_()
indices, counts, means, inverse = pool_points(**case.inputs)
means.sum().backward()
case.inputs['features'].grad, inverse, counts

# %%
# Step 5 — Study complete grouping-loop, unique/scatter, and sorted-segment
# references in the eval script. Which saves Python work? Why can GPU reduction
# ordering change rounding? Each stores points and occupied cells, O(P*C+V*C),
# and never allocates a full grid. Backward-retained state is additional.
# benchmark_pooling() is opt in: seed, workload/device/dtype, repeated timing
# and numerical error are recorded. Correctness has no speed threshold.
# Extension: max-pool with explicit tie gradients; then compare voxel means
# against count-weighted z collapse and direct point-to-pillar pooling.

# %%
# Boundary precision challenge: -tiny is strictly inside a box ending at zero,
# but (-tiny - -1) may round to 1. Membership must be checked in physical
# coordinates first. Only then clip a rounded cell to grid_shape-1; clipping
# before membership would incorrectly admit zero and positive outside rows.
# The near_zero_upper_float32/float64 eval cases have literal expected groups.
tiny=torch.finfo(torch.float32).tiny
probe=torch.tensor([-tiny,0.,tiny],dtype=torch.float32)
probe<0,torch.floor(probe+1)
