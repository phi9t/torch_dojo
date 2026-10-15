# %%
# 0089 · Train a scalar reward model from preferences
# Original dojo exercise motivated by https://www.deep-ml.com/problems/484
# A reward model assigns one scalar score to each complete response. This
# pairwise objective trains those scores; it does not update a policy with DPO.

# %%
import torch
from torch_judge import check

# %%
# Predict before running: if chosen=2 and rejected=0, the zero-margin loss
# for that one pair is log(1 + exp(-2)), about 0.127. The chosen score should
# get a negative gradient so gradient descent raises it. At a tie, loss is
# log(2) and strict preference accuracy is zero.
print('one-pair loss:', torch.nn.functional.softplus(torch.tensor(-2.)).item())

# %%
# Accept chosen_scores and rejected_scores as finite, matching, nonempty
# floating [N] tensors on the same device and with the same dtype. Each row
# is one preferred/rejected response pair. Return two scalar tensors:
#   loss = mean_i softplus(margin - (chosen_i - rejected_i))
#   accuracy = mean_i 1[chosen_i > rejected_i]
# The finite, nonnegative margin changes the loss only. Accuracy uses raw
# scores, counts ties as incorrect, and has no gradient. The loss must keep
# gradients to both score vectors. Do not mutate either input.

# %% [answer]
def pairwise_reward_loss(chosen_scores: torch.Tensor,
                         rejected_scores: torch.Tensor,
                         margin: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mean pairwise logistic loss and strict preference accuracy."""
    pass

# %%
# Baseline: zero margin, mean reduction, strict accuracy, and gradients.
check('0089_pairwise_reward_eval.py', stage='baseline')

# %%
# Follow-up: a positive margin makes the same gap more costly. A tie remains
# incorrect even when the margin changes.
check('0089_pairwise_reward_eval.py', stage='followup')

# %%
# Production: reject malformed inputs with ValueError, including empty or
# mismatched vectors, nonfinite scores, and an invalid margin. Use stable
# softplus for extreme wrong-way gaps, keep finite gradients, and preserve
# the input tensors. Later stages include every earlier check.
check('0089_pairwise_reward_eval.py', stage='production')
