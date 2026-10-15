# %%
# 0103 · Multi-camera projection: find where the point cloud can be sampled
# Baseline: 30–45 minutes. Keep revising the single answer cell.
# This extends 0044 with batched masks; its strict projection API stays unchanged.

# %%
import torch
from torch_judge import check, get_eval_data, compare, replay

# %%
# Step 1 — Name the frames before writing matrix multiplication.
# Ego: x forward, y left, z up. Camera: x right, y down, z forward.
# Units: meters; points are rows. T_camera_from_ego maps ego -> camera:
# camera_xyz = ego_xyz @ R.T + t. K maps camera_xyz to homogeneous pixels.
# uv identifies INTEGER PIXEL CENTERS, not pixel corners.
# Predict where [2,0,0], [2,-2,-2], and [-1,0,0] land under this rotation.
R = torch.tensor([[0.,-1.,0.],[0.,0.,-1.],[1.,0.,0.]])
probe = torch.tensor([[2.,0.,0.],[2.,-2.,-2.],[-1.,0.,0.]])
probe @ R.T

# %%
# Step 2 — Contract (axes never disappear, even at B=M=1):
# points [B,N,3]; K [B,M,3,3]; T_camera_from_ego [B,M,4,4];
# image_hw int64 [B,M,2], HEIGHT then WIDTH; point_valid bool [B,N];
# camera_valid bool [B,M]. B,M positive; N may be zero.
# Return uv [B,M,N,2], camera-z depth [B,M,N], valid bool [B,M,N].
# Support requires both masks, depth > min_depth, and inclusive center bounds
# 0<=u<=W-1, 0<=v<=H-1. Zero BOTH uv and depth when unsupported.
# Behind-camera points are masked observations, not an invalid whole call.
# This only means PROJECTABLE. Without a depth observation, a foreground
# object may still occlude the point. Do not call this an occlusion test.

# %% [answer]
def project_points(points, K, T_camera_from_ego, image_hw, point_valid,
                   camera_valid, min_depth=0.001, chunk_size=None):
    pass

# %%
check('0103_multi_camera_project_eval.py', stage='baseline')

# %%
# Step 3 — Inspect the hand scene. Fix a reversed transform or swapped uv axes
# using individual point predictions, rather than staring at a final score.
eval_data = get_eval_data('0103_multi_camera_project_eval.py')
case = eval_data['optical_axis_and_edges']
case.inputs, case.expected

# %%
actual = project_points(**case.inputs)
result = compare(case, actual)
result.passed, result.diagnostics, result.actual, result.expected

# %%
# Keep result as evidence. Edit and rerun the answer cell, then fetch FRESH
# inputs with replay: changing fixture inputs in place makes comparisons stale.
results = replay('0103_multi_camera_project_eval.py', project_points,
                 names=['optical_axis_and_edges','ninety_degree_camera_turn'])
failures = {name:r for name,r in results.items() if not r.passed}
failures

# %%
# Step 4 — Multiple batches/cameras, translations, 90-degree rotations, and
# missing views. Does changing one camera accidentally change another batch?
# Last row/column are supported pixel centers. Depth == min_depth is excluded;
# compare in the input dtype (0.001 has a different float32 representation).
check('0103_multi_camera_project_eval.py', stage='followup')

# %%
# Step 5 — Reliability: common float32/float64 dtype/device; finite active
# points; bool masks; positive int64 sizes; finite invertible pinhole K with
# last row [0,0,1], positive fx/fy; rigid homogeneous T with det(R)=+1.
# Structure validation permits 1e-5 absolute rounding tolerance. Invalid
# metadata raises ValueError. min_depth finite positive, chunk_size positive
# integer or None. Missing cameras still carry FINITE calibration placeholders.
# Padded points may be NaN: sanitize BEFORE arithmetic, not only at the end.
# Preserve inputs, strides, empty N, and smooth interior gradients through
# points/K/T. No derivative promised through membership/cutoff changes.
check('0103_multi_camera_project_eval.py', stage='production')

# %%
# Step 6 — Predict the gradient before running backward: u=x_camera/z_camera.
# Make an interior point a leaf. Inspect live outputs before compare snapshots
# detach them. The empty/all-missing output must remain zero and mask false.
case = get_eval_data('0103_multi_camera_project_eval.py')['masked_nan_padding']
case.inputs['points'].requires_grad_()
uv, depth, valid = project_points(**case.inputs)
(uv.sum()+depth.sum()).backward()
case.inputs['points'].grad

# %%
# Step 7 — Compare complete per-camera, broadcast, and chunked references in
# the eval script. All qualify production; they trade dispatch and temporaries.
# Chunking bounds FORWARD geometric work, not the M*N required output or saved
# backward state. benchmark_projection() is opt in and reports timing/error;
# correctness has no speed threshold. Try chunk sizes 1, 128, and N.
# Extension: add a measured-depth visibility gate with an explicit tolerance;
# distinguish unknown depth from a genuine occlusion rejection.
