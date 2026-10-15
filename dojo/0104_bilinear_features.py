# %%
# 0104 · Camera pixels to dense feature values
# Baseline timebox: 30–45 minutes. Keep one answer and deepen it in place.
# Projected image uv is not automatically a feature-grid coordinate. Your
# mission: inspect the transform, predict the four corners, sample, then find
# the first counterexample to your current implementation.

# %%
import torch
from torch_judge import check, get_eval_data, compare, replay

# %%
# Inputs:
# features [B,M,C,Hf,Wf], uv [B,M,N,2] in image pixels (u=x,v=y),
# valid bool [B,M,N], feature_from_image [B,M,3,3] affine.
# Float tensors share float32/float64 dtype and device; B,M,C,Hf,Wf > 0.
# Map [x_feature,y_feature,1] = affine @ [u_image,v_image,1].
# The bottom affine row is exactly [0,0,1], its 2x2 linear block invertible,
# and metadata is finite even for unavailable views. Never infer alignment
# from feature/image dimensions: a crop can change offset as well as scale.
# Return (samples [B,M,N,C], supported bool [B,M,N]).
# Integer grid indices are feature centers. Bounds are inclusive:
# 0 <= x_feature <= Wf-1, 0 <= y_feature <= Hf-1. Outside is zero/false,
# with no padding blended into a supposedly supported value.
# Baseline: one view, one channel. Predict x+10*y at (0.25,0.5).
numbered = torch.tensor([[0.,1.,2.],[10.,11.,12.]])
corner_values = numbered[[0,0,1,1],[0,1,0,1]]
weights = torch.tensor([0.75*0.5,0.25*0.5,0.75*0.5,0.25*0.5])
corner_values, weights, (corner_values*weights).sum()

# %% [answer]
def sample_features(features, uv, valid, feature_from_image, chunk_size=None):
    pass  # Replace this function; keep helpers inside this same answer cell.

# %%
check('0104_bilinear_features_eval.py',stage='baseline')

# %%
# Retrieve concrete fixtures without loading the reference implementation.
# Start with the crop/resize case: explain why image (3,4) maps to (0.5,0.5).
eval_data = get_eval_data('0104_bilinear_features_eval.py')
case = eval_data['crop_resize']
case.purpose, case.inputs, case.expected

# %%
actual = sample_features(**case.inputs)
result = compare(case,actual)
# Keep this result: it records the old output even after you edit the answer.
result.passed, result.diagnostics, result.actual, result.expected

# %%
# Follow-up: rectangular grids, many views/channels/batches, singleton H or W.
# At the last center use x1=min(x0+1,Wf-1), and dx=x-x0; do not use
# (x1-x) as an unnormalized weight when x1==x0. Why does that lose a corner?
check('0104_bilinear_features_eval.py',stage='followup')

# %%
# Broken-example challenge: swap x and y in the flattened gather index.
# Which named rectangular case exposes that error before a model is trained?
case = eval_data['rectangular_views']
case.inputs['features'].shape, case.inputs['uv'], case.expected

# %%
# Production: N may be zero; preserve strides, dtype/device and all inputs.
# Reject malformed shapes/dtypes, nonpositive/noninteger chunk_size, invalid
# affine metadata, and nonfinite uv marked valid, with ValueError.
# Policy: if any incoming valid is True for a camera, its entire feature map
# must be finite. An entirely unavailable camera may have NaN features/uv.
# Sanitize masked inputs BEFORE coordinate transforms and gathers. Unsupported
# outputs are zero, and inactive entries have zero, finite backward gradients.
# Interior uv, features and affine coefficients remain differentiable; hard
# support and floor indices are discrete. Gradchecks avoid knots and borders.
# chunk_size optionally bounds per-chunk point interpolation work; required
# output still grows with N. Backward retains state across all chunks.
check('0104_bilinear_features_eval.py',stage='production')

# %%
# Revise the answer cell, rerun it, and replay fresh fixtures. Your saved result
# above remains available for comparing the old failure with the new outcome.
results = replay('0104_bilinear_features_eval.py',sample_features,
                 names=['crop_resize','singleton_1x1','masked_camera_nan','empty_points'])
failures = {name:r for name,r in results.items() if not r.passed}
failures

# %%
# Inspect live gradients separately from detached comparison records.
case = get_eval_data('0104_bilinear_features_eval.py')['interior_gradient']
samples,supported = sample_features(**case.inputs)
grad_features,grad_uv = torch.autograd.grad(samples.sum(),
                                           (case.inputs['features'],case.inputs['uv']))
# For x+10*y, each interior coordinate derivative should be [1,10].
grad_uv, grad_features, supported

# %%
# Open the eval script and run the three complete reference cells one at a
# time: four-corner teaching loop → vectorized gathers → point chunks.
# Compare their outputs before invoking sampling_experiment() explicitly.
# Timing is an observation, never a correctness threshold. Explain temporary
# corner storage vs required output vs autograd-retained state.
# Later extension: derive grid_sample normalization for align_corners=True
# and singleton dimensions, then show parity ONLY with this support contract.
