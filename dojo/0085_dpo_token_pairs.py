# %%
# 0085 · Preference pairs, starting from token logits
# Original dojo exercise based on the DPO objective in
# https://arxiv.org/abs/2305.18290
# Start with 0037_dpo_loss.py if the four sequence log-probabilities are new.

# %%
import torch
import torch.nn.functional as F
from torch_judge import check

# %%
# Two completions answer the same prompt. Their token logits are already
# aligned with the target tokens: logits[b,t] predicts tokens[b,t].
# The boolean mask selects COMPLETION tokens only; prompt and padding are
# excluded. Sum selected token log-probabilities to score each completion.
# Predict the chosen completion's log-probability in this tiny example.
toy_logits = torch.tensor([[[0., 0.], [1., 0.]]])
toy_tokens = torch.tensor([[-100, 0]])
toy_mask = torch.tensor([[False, True]])
print('completion token log-prob:', F.log_softmax(toy_logits[:, 1], -1)[0, 0].item())

# %%
# DPO compares the policy's chosen-minus-rejected sequence gap with the
# frozen reference's gap. Minimize softplus(-beta * (policy_gap-ref_gap)).
# A larger policy gap in favor of the chosen completion lowers the loss.
# Think through what happens when both gaps are identical.

# %% [answer]
def dpo_token_pairs(policy_chosen_logits: torch.Tensor,
                    policy_rejected_logits: torch.Tensor,
                    reference_chosen_logits: torch.Tensor,
                    reference_rejected_logits: torch.Tensor,
                    chosen_tokens: torch.Tensor, rejected_tokens: torch.Tensor,
                    chosen_mask: torch.Tensor, rejected_mask: torch.Tensor,
                    beta: float = 0.1) -> torch.Tensor:
    """Return mean pair loss from aligned [B,T,V] logits and completion masks."""
    pass

# %%
# Baseline: one completion token per side, then multiple preference pairs.
check('0085_dpo_token_pairs_eval.py', stage='baseline')

# %%
# Follow-up: masks hide prompt and padding positions, whose token IDs may be
# -100. Only valid completion positions need in-range IDs. Keep gradients in
# both policy branches; detach both reference branches.
check('0085_dpo_token_pairs_eval.py', stage='followup')

# %%
# Production: all four logits share [B,T,V], floating dtype and device;
# V>=2, T>=1. Each nonempty pair needs at least one chosen/rejected token.
# Reject invalid shapes, masks, IDs, nonfinite logits and beta<=0 with
# ValueError. An empty batch returns graph-connected zero. Use stable
# log_softmax/softplus; no per-vocabulary Jacobian or input mutation.
check('0085_dpo_token_pairs_eval.py', stage='production')
