# %%
# 0113 · Let a BEV query ask the cameras
# Baseline timebox: 35–45 minutes. One height anchor, one camera, then grow.
# Copy your completed projection/sampling helpers into the answer section;
# this script stays independent of other numbered scripts.
# Geometry supplies where to look; the query supplies which features to weight.

# %%
import torch
from torch_judge import check, get_eval_data, compare, replay

# %%
# Inputs: queries [B,Q,C], camera image_features [B,M,C,Hf,Wf], ego reference
# points [B,Q,A,3], K [B,M,3,3], T_camera_from_ego [B,M,4,4], int64 image_hw
# [B,M,2], bool camera_valid [B,M], affine feature_from_image [B,M,3,3].
# B,M,C,Hf,Wf,A positive; Q may be zero. Common float32/float64 and device.
# Finite queries/reference points/calibration. Missing-camera feature maps may
# be NaN; maps with incoming projected support must be finite. Never mutate.
# Ego x-forward,y-left,z-up -> camera x-right,y-down,z-forward: T direction is
# explicit. Project/sampling support includes integer pixel-center bounds and
# camera-z > finite positive min_depth. Feature mapping is supplied explicitly.
# Support is geometric, NOT an occlusion test, depth prediction, or learned lift.
# Keys flatten camera-major then anchor-major. Scores = dot(query,key)/sqrt(C).
# Use valid sampled features as BOTH keys and values. True means eligible.
# No learned projections, bias, dropout, or residual are part of this operator.
# Return updates [B,Q,C], supported [B,Q], and the diagnostic dictionary below.
# All-masked query: zero update/false support/zero weights; never softmax NaNs.
# Use float64 working scores/weights; updates/samples/coordinates keep input
# dtype. Eligible dot scores that exceed float64 must raise ValueError rather
# than claim arbitrary precision. Malformed geometry/layout also raises ValueError.
# Diagnostics (required): uv [B,M,Q,A,2], depth/projected [B,M,Q,A],
# samples [B,M,Q,A,C], sample_valid [B,M,Q,A], scores/weights [B,Q,M,A].
# Projected/sample_valid are boolean. Scores at ineligible keys are zero; the
# explicit masks determine eligibility, not score magnitude. Scores/weights
# are float64. Chunk_size optional positive integer bounds query work only.
eval_data=get_eval_data('0113_bev_cross_attention_eval.py')
case=eval_data['one_key']
case.inputs,case.expected

# %%
# Predict the one-key update: ramp features at pixel (.25,.25) -> [.75,.5].
# With exactly one key, changing the query cannot change its weight of one.

# %% [answer]
def bev_attention(queries,image_features,reference_points,K,T_camera_from_ego,
                  image_hw,camera_valid,feature_from_image,min_depth=.001,chunk_size=None):
    pass  # Keep this implementation and local geometry/sampling helpers here.

# %%
actual=bev_attention(**case.inputs)
first_result=compare(case,actual)
first_result.passed,first_result.diagnostics,first_result.actual,first_result.expected

# %%
check('0113_bev_cross_attention_eval.py',stage='baseline')

# %%
# Follow-up: multiple cameras and heights. Inspect one query's eligibility and
# weights; they should sum to one across supported camera/anchor keys, or zero
# when no key survives. Residual composition (queries+updates) is a separate step.
case=eval_data['camera_height_keys']
case.inputs,case.expected[2]['sample_valid'],case.expected[2]['weights']
check('0113_bev_cross_attention_eval.py',stage='followup')

# %%
# Connect to vision adaptation: this case's maps came from one local supervised
# adapter update on a frozen tiny token backbone. Inspect its explicit lineage.
# No external checkpoint or numbered exercise imports are needed.
case=eval_data['adapted_maps']
case.generation,case.inputs['image_features'],case.expected[0]

# %%
# Production challenge: NaN missing cameras, all-masked rows, huge scores.
# Sanitize before multiplication and replace the all-masked maximum BEFORE
# subtract/exp. Float32 dot products can overflow before a stable softmax even
# begins. Work in float64, and reject nonfinite eligible float64 scores.
case=eval_data['missing_camera_nan']
case.inputs,case.expected[2]['weights']

# %%
# Edit/rerun the answer cell, then fetch fresh inputs; keep your old result.
results=replay('0113_bev_cross_attention_eval.py',bev_attention)
failures={name:r for name,r in results.items() if not r.passed}
failures

# %%
# Backward playground: gradients reach queries, reference coordinates and maps.
# Choose smooth interior points away from depth cutoffs, pixel knots, and masks.
case=get_eval_data('0113_bev_cross_attention_eval.py')['camera_height_keys']
for name in ('queries','reference_points','image_features'):
    case.inputs[name].requires_grad_()
updates,supported,intermediates=bev_attention(**case.inputs)
updates.sum().backward()
case.inputs['queries'].grad,case.inputs['reference_points'].grad,case.inputs['image_features'].grad

# %%
check('0113_bev_cross_attention_eval.py',stage='production')

# %%
# Eval versions: per-query loop, broadcast attention, and complete query chunks.
# All retain geometry/samples/masks/weights for inspection. Chunking bounds
# forward score/corner work, not required diagnostic output or backward state.
# benchmark_attention(inputs) is opt in: workload/error/variability are explicit;
# analytical score bytes are not measured allocator peaks. No speed gate.
# Try permuting cameras and heights; inspect the reordered weights and unchanged
# update. Extension: learned key/value projections, explicit occlusion evidence,
# multiple sampling offsets, or a gated residual are NEW declared contracts.
