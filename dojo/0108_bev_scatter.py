# %%
# 0108 · Give every point its vote in BEV
# Baseline timebox: 30–45 minutes. One answer, progressively deeper challenges.
# Two z voxels land in the same pillar: one summarizes three points, the other
# only one. Averaging the voxel means equally loses the original point mean.

# %%
import torch
from torch_judge import check, get_eval_data, compare, replay

# %%
# indices int64 [K,4] = [batch,z,y,x]; features float32/64 [K,C], C>0;
# counts positive int64 [K]; grid_shape int64 [3] = [X,Y,Z]; batch_size>0.
# All tensors share device. Features are finite; K may be zero. No mask is
# needed here: upstream pooling has already excluded invalid point rows.
# Return (bev [B,C,Y,X], point_counts int64 [B,1,Y,X], occupied bool).
# Collapse z using count-weighted means; repeated exact indices also add.
# y is the BEV row, x its column. Occupancy comes from counts, not features.
# Do not mutate inputs. Reject malformed inputs/ranges/count overflow with
# ValueError BEFORE dense allocation or unsafe integer accumulation.
# No arbitrary memory cap: choose a feasible grid; dense byte/key products
# must fit native indexing/storage limits. Each cell total must fit int64;
# independent cells can have a combined count greater than int64 maximum.
voxel_means = torch.tensor([2.,8.])
voxel_counts = torch.tensor([3,1])
voxel_means.mean(), (voxel_means*voxel_counts).sum()/voxel_counts.sum()

# %% [answer]
def scatter_bev(indices, features, counts, batch_size, grid_shape):
    """Collapse sparse voxels along z into a dense BEV grid of point-count-weighted means.

    Each voxel carries the mean of `count` raw points, so a pillar's feature is the
    count-weighted mean of its voxel means (3.5, not 5, for means 2 and 8 with counts 3 and 1).

    Algorithm:
      1. Validate shapes, counts > 0 and index ranges (ValueError) before allocating.
      2. Flatten (b, y, x) to a pillar id; z is dropped, so voxels differing only in z
         (or repeating an exact index) share a pillar and accumulate.
      3. Sum counts per pillar (int64) -> bev_point_counts.
      4. Weight each voxel by count / pillar_total; weights sum to 1 per pillar.
      5. index_add features * weight into pillars: the sum is already the mean.
      6. Reshape to [B,C,Y,X] / [B,1,Y,X]; occupied = total > 0.

    Properties:
      - count / total <= 1, so the weight is applied before multiplying and
        features * count is never formed.
      - Weights depend only on counts: row order does not change any mean, and
        d bev / d feature_k = count_k / total.
      - Occupancy comes from counts; a zero mean does not imply an empty pillar.
      - Empty pillars stay zero with total 0; no division happens for them.

    Production handling:
      - Everything is validated (ValueError) before any dense allocation: types, dtypes,
        shapes, devices, bool/nonpositive batch_size, nonpositive grid, finite features,
        positive counts, index ranges, and B*Y*X*C and B*Z*Y*X within int64 indexing.
      - Per-pillar count totals are checked for int64 overflow exactly (32-bit halves)
        before the real sum, since index_add_ would wrap silently. Independent pillars may
        each hold up to 2**63-1 points.
      - Widening float32 to float64 and scaling each pillar/channel by its max magnitude keep
        the weighted sum in range; the sum is clamped to [-1, 1] to absorb rounding (seventeen
        float64-max values stay finite). Scale and clamp are detached, so the gradient stays
        count / total.

    Sizing (bytes; K = occupied voxels, P = B*Y*X pillars, s = itemsize of features):
      sparse input   K*(4*8 + C*s + 8)        indices int64 + features + counts int64
      dense output   P*C*s + P*8 + P          bev_features + int64 point counts + bool occupied
      accumulators   P*8 + P*C*s              pillar totals + pillar means (the peak)
      temporaries    K*(8 + s + C*s)          pillar ids, weights, features * weights
      dense voxels   X*Y*Z*C*s                never materialized; the reason to stay sparse
    Worked example: 1M lidar points, 0.075 x 0.075 x 0.2 m voxels over a 108 x 108 x 8 m
    range -> grid 1440 x 1440 x 40 (83M voxels), ~2.5 points per voxel -> K ~ 400k (0.5%),
    P = 2.07M, ~10% of pillars occupied, C = 16 float32, B = 1:
      sparse input ~42 MB | BEV output ~151 MB | peak ~200 MB | dense voxels ~5.3 GB
    Memory scales linearly with B; float64 doubles the C-proportional terms. A smaller
    PointPillars-style grid (0.16 m, 432 x 496) with C = 64 gives a ~55 MB BEV.

    Returns (bev_features [B,C,Y,X], bev_point_counts int64 [B,1,Y,X], occupied bool).
    """
    # Dims: b=batch, k=sparse voxel, c=channel, z/y/x=grid axes,
    #       p=BEV pillar (flattened b*y*x), 4=[batch,z,y,x], 1=singleton
    voxel_inds_k4, voxel_features_kc, voxel_point_counts_k = indices, features, counts

    # Validation first: nothing below allocates a dense tensor or accumulates integers
    tensors = (voxel_inds_k4, voxel_features_kc, voxel_point_counts_k, grid_shape)
    if not all(isinstance(t, torch.Tensor) for t in tensors):
        raise ValueError("tensor inputs required")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive int")
    if voxel_features_kc.ndim != 2 or voxel_features_kc.shape[1] < 1:
        raise ValueError("features must be [K,C] with C > 0")
    num_voxels, num_channels = voxel_features_kc.shape
    if voxel_features_kc.dtype not in (torch.float32, torch.float64):
        raise ValueError("features must be float32 or float64")
    if voxel_inds_k4.dtype != torch.int64 or voxel_inds_k4.shape != (num_voxels, 4):
        raise ValueError("indices must be int64 [K,4]")
    if voxel_point_counts_k.dtype != torch.int64 or voxel_point_counts_k.shape != (num_voxels,):
        raise ValueError("counts must be int64 [K]")
    if grid_shape.dtype != torch.int64 or grid_shape.shape != (3,):
        raise ValueError("grid_shape must be int64 [3]")
    if any(t.device != voxel_features_kc.device for t in tensors):
        raise ValueError("tensors must share a device")
    grid_size_x, grid_size_y, grid_size_z = (int(v) for v in grid_shape.tolist())
    if min(grid_size_x, grid_size_y, grid_size_z) < 1:
        raise ValueError("grid_shape must be positive")
    num_batches = batch_size
    # Python ints cannot overflow: the dense element count must fit native indexing
    num_pillars = num_batches * grid_size_y * grid_size_x
    if num_pillars * num_channels > 2**63 - 1 or num_batches * grid_size_z * grid_size_y * grid_size_x > 2**63 - 1:
        raise ValueError("grid too large for int64 indexing")
    if not torch.isfinite(voxel_features_kc).all():
        raise ValueError("features must be finite")
    if (voxel_point_counts_k <= 0).any():
        raise ValueError("counts must be positive")
    batch_inds_k, z_inds_k, y_inds_k, x_inds_k = voxel_inds_k4.unbind(1)
    if ((batch_inds_k < 0) | (batch_inds_k >= num_batches) | (z_inds_k < 0) | (z_inds_k >= grid_size_z) |
        (y_inds_k < 0) | (y_inds_k >= grid_size_y) | (x_inds_k < 0) | (x_inds_k >= grid_size_x)).any():
        raise ValueError("index out of range")

    # One flat slot per (b, y, x) pillar; z is deliberately dropped
    pillar_inds_k = (batch_inds_k * grid_size_y + y_inds_k) * grid_size_x + x_inds_k

    # Exact int64 totals. index_add_ wraps silently, so split each count into 32-bit
    # halves, sum the halves (they cannot overflow), and test the carry before the real sum.
    # A total fits iff (hi_sum + lo_sum // 2**32) < 2**31; exactly 2**63-1 is allowed.
    low_bits_k = voxel_point_counts_k & 0xFFFFFFFF
    high_bits_k = voxel_point_counts_k >> 32
    zeros_p = torch.zeros(num_pillars, dtype=torch.int64, device=voxel_features_kc.device)
    low_sum_p = zeros_p.index_add(0, pillar_inds_k, low_bits_k)
    high_sum_p = zeros_p.index_add(0, pillar_inds_k, high_bits_k)
    if ((high_sum_p + (low_sum_p >> 32)) >= 2**31).any():
        raise ValueError("pillar point count overflows int64")
    pillar_totals_p = zeros_p.index_add(0, pillar_inds_k, voxel_point_counts_k)

    # Count-weighted mean without overflow. Widen float32. Weights are count/total (<= 1),
    # applied before any multiplication. Scale each pillar/channel by its max magnitude so
    # the weighted sum stays in [-1, 1] up to rounding, clamp that rounding, restore scale.
    # The scale and the clamp correction are detached, so d bev / d feature_k = count_k/total.
    work_features_kc = voxel_features_kc.double()
    pillar_weights_k = voxel_point_counts_k.double() / pillar_totals_p[pillar_inds_k].double()
    with torch.no_grad():
        max_abs_pc = torch.zeros(num_pillars, num_channels, dtype=torch.float64, device=work_features_kc.device)
        max_abs_pc = max_abs_pc.scatter_reduce(
            0, pillar_inds_k[:, None].expand_as(work_features_kc), work_features_kc.abs(),
            reduce="amax", include_self=True)
        safe_max_abs_pc = torch.where(max_abs_pc > 0, max_abs_pc, torch.ones_like(max_abs_pc))
    bounded_kc = work_features_kc / safe_max_abs_pc[pillar_inds_k]
    bounded_mean_pc = torch.zeros(num_pillars, num_channels, dtype=torch.float64, device=work_features_kc.device)
    bounded_mean_pc = bounded_mean_pc.index_add(0, pillar_inds_k, bounded_kc * pillar_weights_k[:, None])
    bounded_mean_pc = bounded_mean_pc + (bounded_mean_pc.clamp(-1, 1) - bounded_mean_pc).detach()
    pillar_means_pc = (bounded_mean_pc * safe_max_abs_pc).to(voxel_features_kc.dtype)

    bev_features_bcyx = pillar_means_pc.view(num_batches, grid_size_y, grid_size_x, num_channels).permute(0, 3, 1, 2)
    bev_point_counts_b1yx = pillar_totals_p.view(num_batches, 1, grid_size_y, grid_size_x)
    bev_occupied_b1yx = bev_point_counts_b1yx > 0
    return bev_features_bcyx, bev_point_counts_b1yx, bev_occupied_b1yx


# %%
check('0108_bev_scatter_eval.py',stage='baseline')

# %%
# Fetch a real fixture: identify the zero-valued OCCUPIED cell.
eval_data = get_eval_data('0108_bev_scatter_eval.py')
case = eval_data['unequal_z_collision']
case.purpose, case.inputs, case.expected

# %%
actual = scatter_bev(**case.inputs)
result = compare(case,actual)
result.passed, result.diagnostics, result.actual, result.expected

# %%
# Deliberate bug: overwrite instead of accumulate. Which point votes vanish?
broken = torch.zeros(1,1,2,3,dtype=torch.float64)
for row,value in zip(case.inputs['indices'],case.inputs['features']):
    b,z,y,x = row.tolist()
    broken[b,:,y,x] = value
broken, case.expected[0]

# %%
# Follow-up: duplicate indices, collisions across z, unequal counts, batches,
# empty sparse inputs. Reverse the row order: should any pillar mean change?
# Explain why a zero feature vector does not imply an empty location.
check('0108_bev_scatter_eval.py',stage='followup')

# %%
# Production: strided features, gradients, exact count totals, extreme finite
# values and no mutation. Widening helps float32, but even float64*count can
# overflow. Normalize count/total BEFORE multiplication. Then inspect seventeen
# float64-max values: rounding in a convex sum can itself overflow. Consider
# per-cell feature scaling and a convex-range correction; state its derivative
# policy. References use detached numerical scaling/correction to retain the
# linear count-weighted derivative. Do not finite-difference beyond float64 max.
check('0108_bev_scatter_eval.py',stage='production')

# %%
# Inspect the failure data; every expected exception is also an eval case.
case = eval_data['count_overflow']
case.inputs, case.expected_exception

# %%
# Revise your answer and rerun that cell, then replay fresh fixtures. The old
# result remains inspectable and is not overwritten by this new result map.
results = replay('0108_bev_scatter_eval.py',scatter_bev,
                 names=['duplicate_indices','huge_float64','max_finite_mean','count_overflow'])
failures = {name:r for name,r in results.items() if not r.passed}
failures

# %%
# Live backward: gradients of a sum of occupied mean features are count/total
# per contributing sparse row, one derivative per channel.
case = get_eval_data('0108_bev_scatter_eval.py')['feature_gradient']
bev,point_counts,occupied = scatter_bev(**case.inputs)
grad_features, = torch.autograd.grad(bev.sum(),case.inputs['features'])
grad_features, point_counts, occupied

# %%
# Compare complete references in the eval script: explicit numerator loop
# (followup domain) → stable flattened index_add → grouped stable reduction.
# Run explore_weight_product_overflow() and bev_experiment() only by choice.
# Grouping can remove dense FLOAT64 accumulator storage; it still returns a
# dense BEV. Separate required output, forward temporaries and saved backward
# state. A speed claim is specific to a measured workload, not a correctness gate.
