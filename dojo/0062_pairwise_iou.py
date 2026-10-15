# %%
# 0062 · Pairwise box IoU, from one overlap to a bounded pair grid
# Goal: Implement `pairwise_iou(left, right)` in NumPy: for $N$ boxes and $M$
# boxes return the $(N, M)$ grid of intersection-over-union values. Keep one
# answer and strengthen it at each stage.
#
# Practice scene: the parking lot. Boxes are axis-aligned footprints in the
# BEV plane (world $x$ and $y$, in metres, seen from above), stored xyxy as
# `[x_min, y_min, x_max, y_max]` with `x_max >= x_min` and `y_max >= y_min`.
# Coordinates are continuous, not pixel indices.

# %%
import numpy as np

# %%
# Baseline — Two parked-car footprints, each of area 4, overlap in a unit square.
left = np.array([[0., 0., 2., 2.]])
right = np.array([[1., 1., 3., 3.]])
overlap_side = np.minimum(left[0, 2:], right[0, 2:]) - np.maximum(left[0, :2], right[0, :2])
print("worked overlap area:", np.prod(np.maximum(overlap_side, 0.)))
# Predict: The IoU of these two boxes, before running this cell.
#
# Why: The union counts the shared square once:
# $$\mathrm{IoU} = \frac{|A \cap B|}{|A| + |B| - |A \cap B|} = \frac{1}{4 + 4 - 1} = \frac{1}{7}$$
#
# Implement `pairwise_iou(left, right)` for `left` $(N, 4)$ and `right`
# $(M, 4)$, returning float $(N, M)$. Broadcast pairs; do not loop over each
# pair. Identical positive-area boxes have IoU 1, and disjoint or touching
# boxes have IoU 0.
# Watch out: Clamp each overlap side at zero BEFORE multiplying. Two
# negative sides multiply to a positive "overlap" for boxes that are far apart.

# %% [drill]
# Far apart, yet overlapping?
# Two unit boxes sit diagonally apart in the BEV plane. Predict the raw
# overlap width, height, their product, and the clamped product.
import numpy as np
a = np.array([0., 0., 1., 1.])
b = np.array([2., 2., 3., 3.])
width = min(a[2], b[2]) - max(a[0], b[0])
height = min(a[3], b[3]) - max(a[1], b[1])
print(width, height, width * height, max(width, 0.) * max(height, 0.))

# %% [drill]
# Continuous metres or inclusive pixels?
# Predict the IoU of `[0,0,2,2]` and `[1,1,3,3]` with continuous widths
# $x_{max} - x_{min}$, then with the pixel habit $x_{max} - x_{min} + 1$.
import numpy as np
def iou(p, q, plus):
    inter_w = max(min(p[2], q[2]) - max(p[0], q[0]) + plus, 0.)
    inter_h = max(min(p[3], q[3]) - max(p[1], q[1]) + plus, 0.)
    inter = inter_w * inter_h
    area = lambda r: (r[2] - r[0] + plus) * (r[3] - r[1] + plus)
    return inter / (area(p) + area(q) - inter)
box_a = np.array([0., 0., 2., 2.])
box_b = np.array([1., 1., 3., 3.])
print(round(iou(box_a, box_b, 0.), 4), round(iou(box_a, box_b, 1.), 4))

# %% [drill]
# Which axis makes the pair grid?
# With 2 left boxes and 3 right boxes, predict the shape of the first
# expression and what the second one does.
import numpy as np
a = np.zeros((2, 4))
b = np.zeros((3, 4))
print(np.maximum(a[:, None, 0], b[None, :, 0]).shape)
try:
    np.maximum(a[:, 0], b[:, 0])
except ValueError as error:
    print("ValueError:", error)

# %% [answer]
# Edit this one cell across all three stages. Helpers may live here too.
def pairwise_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    pass

# %%
from torch_judge import check
check("0062_pairwise_iou_eval.py", stage="baseline")

# %%
# Follow-up — Preserve the baseline. Handle $(0, 4)$ empty inputs, zero-area
# boxes, and transpose symmetry: `pairwise_iou(A, B) == pairwise_iou(B, A).T`.
# A zero-union pair (two zero-area boxes) has IoU 0, not NaN.
# Watch out: Boxes are continuous metres in the BEV plane, so a side is
# $x_{max} - x_{min}$. Adding one "pixel" to every width inflates small boxes
# and changes their IoU.
check("0062_pairwise_iou_eval.py", stage="followup")

# %%
# Production — Reject reversed corners, nonfinite values, and wrong shapes
# with `ValueError`, even when the other input is empty. Do not mutate inputs.
# Translating the whole scene, or scaling it by a positive factor, leaves IoU
# unchanged. Return the required $O(NM)$ output; for large sets, process pair
# blocks so temporary coordinate arrays use $O(\text{block}^2)$ memory, not
# $O(NM)$ extra space. A loop over blocks is fine; a Python loop over every
# box pair is not.
check("0062_pairwise_iou_eval.py", stage="production")
