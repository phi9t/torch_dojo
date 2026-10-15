# %%
# 0106 · A point's neighborhood passport
# Baseline timebox: 30–45 minutes. Keep one evolving answer cell.
# What can a voxel centroid tell a network that raw xyz does not?

# %%
import torch
from torch_judge import check, get_eval_data, compare, replay

# %%
# Inputs: points [B,N,3], attributes [B,N,F], bool point_valid [B,N],
# float origin [3], positive voxel_size [3], positive int64 grid_shape [3].
# Metadata is xyz ordered. Floating inputs share float32/float64 dtype/device;
# masks and grid_shape use the same device. B>0, N may be zero, F may be zero.
# Physical bounds are half-open: origin <= xyz < origin + size*grid_shape.
# Return descriptors [B,N,F+10] and inside [B,N]; preserve order/dtype/device.
# Channel order: attributes | xyz | xyz-centroid | xyz-voxel_center | range.
# Centroids include every valid inside point, including duplicates, separately
# for each batch and voxel. Outside/padded rows are zero and false.
# Valid coordinates must be finite; included attributes must be finite. Masked
# padding and outside attributes may be NaN. Reject malformed metadata with
# ValueError; derived upper bounds must be finite in float64. Never mutate input.
eval_data = get_eval_data('0106_point_decoration_eval.py')
case = eval_data['one_voxel']
case.inputs, case.expected

# %%
# Work the first case by hand: centroid [1,.5,.5], cell center [1,1,1].
# Which descriptor channels change if the first point is duplicated?
# Which channels remain tied to ego coordinates? Write those predictions first.

# %% [answer]
def decorate_points(points, attributes, point_valid, origin, voxel_size, grid_shape):
    pass  # Define your helper functions here and revise this answer as you go.

# %%
actual = decorate_points(**case.inputs)
first_result = compare(case,actual)
first_result.passed, first_result.diagnostics, first_result.actual, first_result.expected

# %%
check('0106_point_decoration_eval.py',stage='baseline')

# %%
# Follow-up: translated origin, anisotropic cell size, batches, duplicates,
# and point permutations. Keep the descriptor in original point order.
# Predict the upper-edge mask before inspecting the boundary_rows case.
case = eval_data['batch_duplicate']
case.inputs, case.expected
check('0106_point_decoration_eval.py',stage='followup')

# %%
# Production challenge: allocating grid_xyz cells is impossible in this case.
# Group only occupied coordinates, using explicit integer coordinate tuples.
# Multiplying all three grid dimensions into an int64 key can overflow too.
case = eval_data['sparse_huge_grid']
case.inputs, case.expected

# %%
# The range can be finite even when x*x+y*y+z*z overflows. Widen float32
# reductions and rescale the norm; retain float64 when inputs are float64.
# Remove excluded NaN values before arithmetic so their gradients remain zero.
# Floor and membership are discrete; gradients are tested only with FIXED
# membership away from cell boundaries and range=0.
case = eval_data['wide_range']
case.inputs, case.expected

# %%
# Rerun the edited answer cell, then fresh replay; your old result stays intact.
results = replay('0106_point_decoration_eval.py',decorate_points,
                 names=['batch_duplicate','masked_padding','wide_range'])
failures = {name:r for name,r in results.items() if not r.passed}
failures

# %%
# Inspect how changing one point changes BOTH centroid offsets in its voxel.
case = get_eval_data('0106_point_decoration_eval.py')['one_voxel']
case.inputs['points'].requires_grad_()
descriptors,inside = decorate_points(**case.inputs)
descriptors[0,0,4].backward()  # F=1: channel 4 is the first centroid offset.
case.inputs['points'].grad  # Expected first-coordinate derivatives [.5,-.5].

# %%
check('0106_point_decoration_eval.py',stage='production')

# %%
# In the eval script, compare decorate_loop / decorate_grouped / decorate_stable
# as complete runnable versions. Run explore_range_overflow() deliberately.
# benchmark_decoration() reports the chosen workload/error/timing variability;
# small CPU examples need not favor the more vectorized version.
# Extensions: replace centroid by a robust statistic; add normalized offsets;
# explain how moving a point across a cell boundary changes the descriptor.

# %%
# Boundary precision challenge: -tiny is strictly inside a box ending at zero,
# but (-tiny - -1) may round to 1. Membership must be checked in physical
# coordinates first. Only then clip a rounded cell to grid_shape-1; clipping
# before membership would incorrectly admit zero and positive outside rows.
# The near_zero_upper_float32/float64 eval cases have literal expected groups.
tiny=torch.finfo(torch.float32).tiny
probe=torch.tensor([-tiny,0.,tiny],dtype=torch.float32)
probe<0,torch.floor(probe+1)
