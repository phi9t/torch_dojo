# %%
# 0070 · Anchor to GT matching
# Goal: Implement `match_anchors(iou, negative_threshold, positive_threshold)`
# in NumPy: label every anchor as negative, ignored, or matched to one ground
# truth (GT). Revise the same answer cell as you advance through all three stages.
#
# Practice scene: the parking lot. Anchors are fixed prior boxes tiled over the
# BEV plane (world $x$ and $y$, in metres); GTs are the parked-car footprints.
# The input is the 0062 grid `pairwise_iou(anchors, gts)`, so it is
# $(A, G)$: row $a$ is an anchor, column $g$ a GT. No coordinates are needed here.
#
# Authored practice variant; see `practice_bank/drills.tsv` for evidence.

# %%
import numpy as np

# %%
# Baseline — Find each anchor's best GT along its row.
probe_iou = np.array([[.2, .6], [.4, .1], [.7, .2]])
print("best GT per anchor:", probe_iou.argmax(axis=1))
# Return integer labels $(A,)$: $-1$ negative, $-2$ ignored, otherwise a GT
# index. Without forced matches, a row maximum below `negative_threshold` is
# negative, one at or above `positive_threshold` is positive, and the gap is
# ignored.
# Watch out: Equality at the lower threshold is ignored, equality at the upper
# threshold is positive: $\text{IoU} < 0.3$ is negative and $\text{IoU} \ge 0.5$
# is positive.
#
# Predict: Labels for the probe with thresholds 0.3 and 0.5, before coding.

# %% [drill]
# Rows or columns?
# Predict the best GT for each anchor and the best anchor for each GT, and
# each result's shape.
import numpy as np
iou = np.array([[.2, .6], [.4, .1], [.7, .2]])
print(iou.argmax(axis=1), iou.argmax(axis=0))

# %% [drill]
# On the threshold
# Three anchors' best IoUs sit at and just under the thresholds 0.3 and 0.5.
# Predict the labels, each anchor's best GT being 0.
import numpy as np
best = np.array([.3, .5, .29])
labels = np.full(3, -2)
labels[best < .3] = -1
labels[best >= .5] = 0
print(labels)

# %% [drill]
# Two GTs want the same anchor
# Predict each GT's best anchor, each anchor's best GT, and each anchor's best
# IoU. What should anchor 1 end up as once forced matching runs?
import numpy as np
iou = np.array([[.9, .8], [.1, .2]])
print(iou.argmax(axis=0), iou.argmax(axis=1), iou.max(axis=1))

# %% [answer]
def match_anchors(iou: np.ndarray, negative_threshold: float = .3,
                  positive_threshold: float = .5) -> np.ndarray:
    pass

# %%
from torch_judge import check
check("0070_anchor_match_eval.py", stage="baseline")

# %%
# Follow-up — Force the best anchor for each GT (its column maximum), even
# when that IoU is below the positive threshold, so every parked car gets an
# anchor. Column ties choose the lowest anchor. With $G = 0$ all anchors are
# negative; with $A = 0$ return an empty $(0,)$ array.
# Watch out: A collision is won by the lowest GT index, and the losing GT does
# not claim its second-best anchor. This is deterministic many-to-one matching,
# not a global assignment.
check("0070_anchor_match_eval.py", stage="followup")

# %%
# Production — Require finite IoU in $[0, 1]$, shape $(A, G)$, and thresholds
# $0 \le \text{negative} \le \text{positive} \le 1$. Do not mutate the IoU. A
# row's ordinary best-GT tie chooses the lowest index. The output costs $O(A)$;
# no pairwise geometry is needed because the IoU is already supplied.
check("0070_anchor_match_eval.py", stage="production")
