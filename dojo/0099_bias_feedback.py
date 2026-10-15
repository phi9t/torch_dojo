# %%
# 0099 · The router learns from yesterday's traffic
# A persistent bias nudges future *selection* toward underused experts.
# DeepSeek-V3 describes a whole-batch controller; Olmo-core 3 discusses
# the same sign-feedback mechanism. This is an original local exercise.
# https://arxiv.org/html/2412.19437#S2.SS1.SSS2
# https://allenai.org/papers/olmocore3

# %%
import torch
from torch import nn
from torch_judge import check

# %%
# Example: with three experts and four requests all going to expert 0,
# counts=[4,0,0], mean=4/3. After one completed training batch, update
# b[e] += gamma * sign(mean-counts[e]). What is the new bias for gamma=.1?
# Nothing updates while a microbatch is still routing.
print('requests [4, 0, 0] mean:', 4 / 3)

# %% [answer]
class BiasFeedback(nn.Module):
    def __init__(self, num_experts: int, step_size: float):
        super().__init__()
        pass

    def forward(self, logits: torch.Tensor, k: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        """Return selected expert IDs and unbiased normalized affinities."""
        pass

    def finish_batch(self) -> None:
        """Update bias once from accumulated requests, then clear counts."""
        pass

# %%
# Baseline: register persistent buffers bias[E] (floating zeroes) and
# counts[E] (long zeroes). For logits[T,E], use sigmoid(logits) as the
# expert affinities. Select top-k by affinity+bias, stable ties favoring
# the lower expert ID. Every selected (token, expert) pair adds one count.
# finish_batch updates bias by the rule above and resets counts.
check('0099_bias_feedback_eval.py', stage='baseline')

# %%
# Follow-up: two calls to forward are two microbatches of ONE optimizer
# batch. Accumulate both before calling finish_batch; updating after each
# would change the next microbatch's route choices and feedback signal.
# Calling finish_batch with no pending routes changes nothing.
check('0099_bias_feedback_eval.py', stage='followup')

# %%
# Production: return weights from the chosen *unbiased* affinities,
# normalized over those k choices. Bias changes which experts run, but
# must never appear in these output coefficients. Keep bias/count state
# outside autograd and retain gradients through selected affinities.
# Empty T returns [0,K] IDs/weights and no pending updates; equal counts
# cause no change. Reject non-finite logits, bad width/dtype/k and
# invalid constructor values with ValueError. Bias is a feedback
# controller, not a guarantee that all future route histograms balance.
# For very negative logits, normalize selected log-sigmoid affinities
# stably; direct sigmoid division can become 0/0. If two affinities
# underflow to zero before top-k, preserve their original logit order.
check('0099_bias_feedback_eval.py', stage='production')
