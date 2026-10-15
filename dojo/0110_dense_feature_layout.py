# %%
# 0110 · Put spatial tokens back on the map
# Baseline timebox: 30 minutes. One editable function grows across stages.
# A square-root guess is not a feature-grid contract. Keep height and width.

# %%
import torch
from torch_judge import check, get_eval_data, compare, replay

# %%
# Input tokens [B,P+Hf*Wf,C], explicit positive integer grid_hw=(Hf,Wf),
# and nonnegative integer prefix_tokens=P. C>0; B may be zero.
# Return [B,C,Hf,Wf], row-major spatial order, exactly excluding P prefix tokens.
# Preserve float32/float64 dtype/device and gradients. Reject invalid layout
# metadata, mismatched length, and malformed inputs with ValueError.
# No input mutation. Output may be a view: do not mutate it if aliases matter.
# This exercise rearranges values; it makes no camera-coordinate mapping claim.
# Supply feature_from_image separately when connecting the maps to projection.
eval_data = get_eval_data('0110_dense_feature_layout_eval.py')
case = eval_data['square_numbered']
case.inputs, case.expected

# %%
# Prediction: each token has two channels. Write each channel's 2x2 grid.
# What error would reshape(B,C,H,W) produce without moving the channel axis?

# %% [answer]
def dense_features(tokens,grid_hw,prefix_tokens=0):
    pass  # Improve this single implementation; helpers belong in this section.

# %%
actual = dense_features(**case.inputs)
first_result = compare(case,actual)
first_result.passed, first_result.diagnostics, first_result.actual, first_result.expected

# %%
check('0110_dense_feature_layout_eval.py',stage='baseline')

# %%
# Follow-up: two prefix tokens and a 2x3 rectangular map, then an empty batch.
# Prefixes may be class/register tokens; their meaning is outside this layout
# operation. Only the supplied prefix count tells us which tokens to exclude.
case = eval_data['rectangular_prefix']
case.inputs, case.expected
check('0110_dense_feature_layout_eval.py',stage='followup')

# %%
# Production: noncontiguous BLC tensors and gradients, without silent copies
# being mistaken for free views. reshape may allocate; permute changes strides.
# A contiguous copy can help a downstream kernel but costs storage/bandwidth.
case = eval_data['noncontiguous']
case.inputs['tokens'].stride(), case.expected

# %%
# Edit/rerun your answer cell, then compare fresh cases. Old first_result stays
# available even when the new implementation passes.
results = replay('0110_dense_feature_layout_eval.py',dense_features)
failures = {name:r for name,r in results.items() if not r.passed}
failures

# %%
# Prefix gradient should be exactly zero; spatial gradient follows the layout.
case = get_eval_data('0110_dense_feature_layout_eval.py')['rectangular_prefix']
case.inputs['tokens'].requires_grad_()
feature_map = dense_features(**case.inputs)
feature_map.sum().backward()
case.inputs['tokens'].grad

# %%
check('0110_dense_feature_layout_eval.py',stage='production')

# %%
# Compare complete dense_loop / dense_view / dense_contiguous eval cells.
# inspect_layout_storage() shows strides and actual storage aliasing.
# benchmark_layout() is opt in and reports chosen workload/error/timing
# variability. Neither reports an allocator peak or promises universal speed.
# Next: position-grid interpolation and a frozen backbone's trainable adapter.
