# %%
# 0128 · Pairwise box IoU grid, workout contract (practice set 4/7, 15 min)
# Goal: implement `pairwise_iou` fully vectorized. A stricter, validated
# version of this problem is 0062; this one uses the contract stated below.

# %%
import numpy as np

# %%
# Baseline — boxes_a[N,4], boxes_b[M,4], ordered [xmin,ymin,xmax,ymax],
# continuous coordinates (no +1). Return [N,M].
# Touching boundaries have zero intersection. Zero-area boxes are allowed;
# if the union area is zero, return IoU 0 (not NaN). N=0 or M=0 keeps its shape.
# Hand-checkable: [0,0,2,2] vs [1,1,3,3] -> 1/7.

# %% [drill]
# Far apart, yet "overlapping"?
# Predict the raw overlap sides and their product for [0,0,1,1] vs [2,2,3,3].
w = min(1, 3) - max(0, 2); h = min(1, 3) - max(0, 2)
print(w, h, w * h, max(w, 0) * max(h, 0))

# %% [drill]
# Gradient of 1 - IoU
# For two DISJOINT boxes, predict IoU, then d IoU / d (box position).
# Why does a detector trained only on 1 - IoU stall when predictions miss the target?

# %%
# Inputs conforming to the answer's signature.
boxes_a: np.ndarray = np.array([[0., 0., 2., 2.], [1., 1., 1., 1.]])             # [N,4]
boxes_b: np.ndarray = np.array([[1., 1., 3., 3.], [5., 5., 6., 6.], [0., 0., 2., 2.]])  # [M,4]

# %% [answer]
def pairwise_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    # boxes_a [N,4], boxes_b [M,4] -> iou [N,M] float64, no Python loop over pairs
    pass

# %%
from torch_judge import check
check("0128_box_iou_grid_eval.py", stage="baseline")

# %%
# Follow-up — empty inputs keep (N,0)/(0,M)/(0,0), results match a scalar
# reference on random boxes, and IoU is within [0, 1].
check("0128_box_iou_grid_eval.py", stage="followup")

# %%
# Production — clamp each overlap side BEFORE multiplying; do not mutate
# inputs; a 1500x1500 grid must finish quickly (no pair loop). State time
# O(NM) and peak intermediate memory O(NM) for the [N,M,2] corner tensors.
# Extension (not graded): mean matched-box loss 1 - IoU with a validity mask.
check("0128_box_iou_grid_eval.py", stage="production")
