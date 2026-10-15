# %%
# 0037 · Direct preference optimization
# One editable function, three increasingly useful contracts. Run cells in
# Emacs/Jupyter and revise the answer in place. Inputs are sequence log-probs,
# not token logits. The objective here is an authored practice contract.

# %%
import torch
import torch.nn.functional as F

# %%
# Baseline: finite [B] policy chosen/rejected and reference chosen/rejected
# log-probs. Return the scalar MEAN of softplus(-beta * advantage), where
# advantage = (policy_chosen-policy_rejected) - (ref_chosen-ref_rejected).
# beta defaults to 0.1. Predict: if both gaps agree, what is the loss?
policy_gap = torch.tensor([2.]) - torch.tensor([1.])
reference_gap = torch.tensor([3.]) - torch.tensor([2.])
print('equal gaps:', policy_gap, reference_gap, 'loss:', F.softplus(-(policy_gap-reference_gap)).item())

# %% [answer]
def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    pass

# %%
from torch_judge import check
check('0037_dpo_loss_eval.py', stage='baseline')

# %%
# Follow-up: reference log-probs are fixed targets. Detach both references,
# but keep policy gradients. A reference gap of +1 and policy gap of +1
# gives log(2), not a zero loss. Explain why batch repetition does not change
# a mean loss.
check('0037_dpo_loss_eval.py', stage='followup')

# %%
# Production: use a stable logistic expression for gaps near +/-1000. Define
# the empty batch as a graph-connected zero. Reject mismatched/nonfinite
# tensors and beta <= 0. The computation needs O(B) temporary storage.
check('0037_dpo_loss_eval.py', stage='production')
