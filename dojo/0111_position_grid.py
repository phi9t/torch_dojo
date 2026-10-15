# %%
# 0111 · Positional grids: rectangular resize without moving special tokens
# Baseline 30–45 minutes. Keep revising the single answer cell.
# Explicit grid sizes beat inferring sqrt(token_count): not every image is square.

# %%
import torch
from torch_judge import check, get_eval_data, compare, replay

# %%
# Step 1 — Derive the half-pixel mapping.
# For destination x, source sx=(x+0.5)*old_W/new_W-0.5, clamped to [0,old_W-1].
# Same for y. Interpolate FOUR neighboring pixel centers (clamp-to-edge).
# Predict resizing [0,2] from width 2 to width 4: [0,0.5,1.5,2].
# align_corners=False aligns outer pixel edges; True would line up end centers
# and produce a different ramp. This exercise never switches conventions.
source_positions = torch.tensor([[[0.],[2.]]],dtype=torch.float64)

# %%
# Step 2 — Contract:
# positions [B,L,C] finite float32/float64, C>0, empty B permitted;
# old_hw/new_hw are tuple/list pairs of positive Python integers (HEIGHT,WIDTH).
# prefix_tokens is a nonnegative Python integer; bool is not valid metadata.
# L=prefix_tokens+old_H*old_W. Spatial order is row-major y then x;
# prefix tokens remain at the front, copied EXACTLY, never interpolated.
# Return [B,prefix_tokens+new_H*new_W,C], original dtype/device.
# Bilinear only, align_corners=False, antialias=False, clamp-to-edge.
# Same-size grids take an EXACT identity path (returning a source alias allowed).
# Invalid shape/count/dtype/nonfinite data/grid metadata raises ValueError.

# %% [answer]
def resize_positions(positions, old_hw, new_hw, prefix_tokens=0):
    pass

# %%
check('0111_position_grid_eval.py',stage='baseline')

# %%
# Inspect the interpolation geometry independently of any checkpoint.
eval_data = get_eval_data('0111_position_grid_eval.py')
case = eval_data['one_by_two_ramp']
case.inputs, case.expected

# %%
actual = resize_positions(**case.inputs)
result = compare(case,actual)
result.passed, result.diagnostics, result.actual, result.expected

# %%
# Save result, edit/rerun the answer, then request fresh inputs. The original
# mismatch remains inspectable. Avoid editing a case in place before compare.
results = replay('0111_position_grid_eval.py',resize_positions,
                 names=['one_by_two_ramp','prefix_rectangular_upsample'])
failures = {name:r for name,r in results.items() if not r.passed}
failures

# %%
# Step 3 — Prefix + rectangular grids, multiple batches, multiple channels,
# upsampling/downsampling and singleton dimensions. Try exchanging H and W:
# equal token counts will not expose that error, but the numbered ramp will.
# Downsampling here does NOT antialias: specify a different contract if the
# application requires low-pass filtering instead of sampling this surface.
check('0111_position_grid_eval.py',stage='followup')

# %%
# Step 4 — Preserve noncontiguous inputs and empty batches; avoid in-place
# updates. Prefix and identity equality must be exact; interpolated values use
# floating tolerance. Compute in source dtype; the separable reference derives
# coordinates in float64 then converts weights to source dtype.
# Gradients must reach source positions, including untouched prefix positions.
check('0111_position_grid_eval.py',stage='production')

# %%
# Inspect live gradients; compare records detach tensor snapshots.
case = get_eval_data('0111_position_grid_eval.py')['prefix_rectangular_upsample']
case.inputs['positions'].requires_grad_()
resized = resize_positions(**case.inputs)
resized.sum().backward()
case.inputs['positions'].grad

# %%
# Step 5 — Compare explicit four-neighbor, separable, and framework versions
# in the eval script. All are complete and qualified production. Interpolation
# parity assumes matching mode/corners/antialias/layout/dtype; it does not
# establish that a particular model trained its positions with this convention.
# Separable work uses [B,old_H,new_W,C] plus output; this may be larger than
# expected when downsampling height. Framework workspace depends on backend.
# Optional benchmark_positions() reports workload, timing variability and error.
# Speed has no correctness threshold; no invented stability gain is claimed.
# Extension: compare align_corners=True and antialiased downsampling visually.
# Then explain how an adapter records a checkpoint's actual interpolation rules.
