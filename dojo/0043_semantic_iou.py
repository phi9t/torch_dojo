# %%
# 0043 · Masked semantic IoU
# Class 0 is undefined. Only support=True pixels with truth != 0 are eligible.
# Average IoU over classes 1..C-1 with nonzero union. Return scalar zero if none.

# %%
import torch

# %%
# Worked probe — predict the output before running this cell.
truth = torch.tensor([1, 1, 2, 0])
pred = torch.tensor([1, 2, 0, 2])
eligible = truth != 0
print("eligible pairs:", list(zip(truth[eligible].tolist(), pred[eligible].tolist())))
# Predict: class 1 IoU=1/2, class 2 IoU=0; mean=1/4.

# %%
# Baseline contract
# Count intersection and union inside the eligible mask. A predicted 0 on
# a labeled pixel is a false negative; unsupported pixels contribute nothing.
# Work through shapes and one boundary case before editing the answer.

# %% [answer]
def semantic_iou(truth: torch.Tensor, prediction: torch.Tensor, support: torch.Tensor, num_classes: int) -> torch.Tensor:
    pass

# %%
from torch_judge import check
check('0043_semantic_iou_eval.py', stage='baseline')

# %%
# Follow-up contract
# A class only predicted on an eligible pixel enters the mean. All-zero truth,
# all-unsupported, and empty arrays return scalar zero. Preserve arbitrary label shape.
# Revise the same answer cell, then run this cumulative check.
check('0043_semantic_iou_eval.py', stage='followup')

# %%
# Production contract
# Validate matching shapes/devices, integer labels in [0,C), boolean support,
# and C>=2. Reject malformed input; do not mutate it. O(P+C) or O(PC) is
# acceptable; state the tradeoff between a confusion matrix and per-class masks.
# This check reruns baseline and follow-up before production verifiers.
check('0043_semantic_iou_eval.py', stage='production')
