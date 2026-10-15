# %%
# 0074 · Detection average precision
# Goal: Compute one-class detection average precision (AP) from scored
# detections that an upstream matcher has already paired with ground truth (GT).
# Authored practice variant; see practice_bank/drills.tsv for related drills.
# Revise the same answer cell as you advance through all three stages.

# %%
import numpy as np

# %%
# Baseline — One class only. Each detection is `(image_id, score, matched_gt_id)`.
# A GT ID of `-1` means no match. IDs are scoped to an image, so `(0,0)`
# and `(1,0)` refer to different objects.
probe_detections = [(0, .9, 0), (0, .8, 0), (0, .7, 1)]
print("duplicate first GT:", probe_detections)
# Sort by descending score and return all-point interpolated AP in $[0,1]$.
# Watch out: The denominator of recall is `total_gt`, including GT that no
# detection ever matched, not the number of true positives found.
#
# Predict: Which of the first two detections is a false positive after sorting?

# %% [drill]
# Same GT ID, different image
# Predict the true-positive flags when claims are keyed by (image, GT) and
# when they are keyed by GT ID alone.
import numpy as np
dets = [(0, .9, 0), (0, .8, 0), (1, .7, 0)]
by_pair, by_gt, pair_tp, gt_tp = set(), set(), [], []
for image, _score, gt in dets:
    pair_tp.append(int((image, gt) not in by_pair)); by_pair.add((image, gt))
    gt_tp.append(int(gt not in by_gt)); by_gt.add(gt)
print(pair_tp, gt_tp)

# %% [drill]
# What does recall divide by?
# Two of four GT objects are found. Predict the recall curve both ways.
import numpy as np
tp = np.array([1, 0, 1])
print(np.cumsum(tp) / 4, np.cumsum(tp) / tp.sum())

# %% [drill]
# A false positive first
# Two GT, and the top-scored detection is a false positive. Predict the raw
# precision, its envelope, and AP with and without the envelope.
import numpy as np
tp = np.array([0, 1, 1])
precision = np.cumsum(tp) / np.arange(1, 4)
envelope = np.maximum.accumulate(precision[::-1])[::-1]
gain = np.diff(np.cumsum(tp) / 2, prepend=0)
print(precision.round(3), envelope.round(3))
print(round(np.sum(gain * precision), 3), round(np.sum(gain * envelope), 3))

# %% [answer]
def detection_ap(detections: list[tuple[int, float, int]], total_gt: int) -> float:
    pass

# %%
from torch_judge import check
check("0074_detection_ap_eval.py", stage="baseline")

# %%
# Follow-up — Score ties preserve input order. The first scored claim of
# each `(image_id, gt_id)` is a TP; repeats and `-1` are FPs. Cumulative
# precision is $\mathrm{TP}/(\mathrm{TP}+\mathrm{FP})$. Form the right-to-left
# precision envelope $\hat p_k = \max_{j \ge k} p_j$ and integrate it over each
# increase in recall:
# $$\mathrm{AP} = \sum_k (r_k - r_{k-1}) \, \hat p_k$$
# An FP before a TP can reduce AP. Empty detections or `total_gt=0` give
# AP $= 0$ by contract.
check("0074_detection_ap_eval.py", stage="followup")

# %%
# Production — Reject negative or noninteger `total_gt`, malformed triples,
# non-finite scores, negative image IDs and GT IDs below `-1`. Do not mutate
# detections. More distinct matched GT than `total_gt` is invalid. Sorting
# costs $O(D \log D)$; matching and the envelope cost $O(D)$.
# A supplied matched_gt_id presumes an upstream matcher; this script does
# not infer boxes, IoU thresholds, or ignored GT policies.
check("0074_detection_ap_eval.py", stage="production")
