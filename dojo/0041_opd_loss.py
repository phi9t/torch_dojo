# %%
# 0041 · On-policy distillation reverse KL
# One editable answer, checked in stages. This exercise compares categorical
# distributions at observed token positions; it does not sample trajectories.

# %%
import torch
import torch.nn.functional as F
from torch import Tensor

# %%
# Baseline: student logits [...,V], one teacher with the same shape.
# Return mean_t sum_v p_student(v) * (log p_student(v)-log p_teacher(v)).
# Use log_softmax. Predict why equal distributions give zero before running.
probe = torch.tensor([[0., 0.]])
print('equal distribution:', probe.softmax(-1))

# %% [answer]
def opd_loss(student_logits: Tensor, teacher_logits: Tensor,
             teacher_weights: Tensor | None = None, mask: Tensor | None = None,
             temperature: float = 1.0) -> Tensor:
    pass

# %%
from torch_judge import check
check('0041_opd_loss_eval.py', stage='baseline')

# %%
# Follow-up: teacher logits may be [T,...,V]. Normalize nonnegative [T]
# weights, then combine per-teacher KLs. A token mask is a nonnegative weight
# array [...]; divide by the SUM OF MASK WEIGHTS, not the batch size. Detach
# teacher logits and weights. Predict the answer for one valid token repeated
# twice with mask weights 1 and 2.
check('0041_opd_loss_eval.py', stage='followup')

# %%
# Production: divide logits by positive temperature, then multiply the
# resulting KL by temperature squared. All-zero mask yields graph-connected
# zero. Reject invalid shapes, nonfinite data, negative weights, zero total
# teacher weight, and nonpositive temperature. Work with O(T * tokens * V)
# memory or less; no dense vocabulary-by-vocabulary Jacobian is needed.
check('0041_opd_loss_eval.py', stage='production')
