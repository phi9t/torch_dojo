# %%
# 0095 · Four ways to weight a routed expert
# Try this after 0028_moe.py. The experts have already produced their
# outputs; here the puzzle is entirely in the router's coefficients.
# Sources: DeepSeek-V2 Eq. 20–22 https://arxiv.org/html/2405.04434#S2.SS2.SSS1
# DeepSeek-V3 Eq. 12–16 https://arxiv.org/html/2412.19437#S2.SS1.SSS2
# Qwen3 compatible forward https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py
# The Qwen link is a public compatible implementation, not its training loop.

# %%
import torch
from torch_judge import check

# %%
# Four experts vote on one token. First predict the result by hand:
# if only two experts run, must their mixture weights still sum to one?
toy_logits = torch.tensor([[2., 1., 0., -1.]])
print('all-expert probabilities:', toy_logits.softmax(-1))
print('selected probability mass:', toy_logits.softmax(-1)[0, :2].sum())
# An expert output is already computed: expert_outputs[t,e,:]. Never
# call an expert network in this workout. The output is the weighted
# sum over selected experts. Return (output [T,D], expert_ids [T,k],
# weights [T,k], router_scores [T,E]).
#
# Modes:
# selected_softmax: rank by raw logits; softmax only selected logits.
# full_truncate: rank by full softmax probabilities; KEEP selected mass.
# full_renorm: rank by full probabilities; divide selected mass by its sum.
# sigmoid_bias: rank by sigmoid(logits)+bias, but weight using the
# UNBIASED selected sigmoid affinities normalized to sum one, then scale.
# For selected_softmax, router_scores are raw logits; for both full
# modes they are full softmax probabilities; for sigmoid_bias they are
# unbiased sigmoid affinities. Bias affects IDs only, never weights.
# Equal ranking scores choose smaller expert IDs first.
# Inputs share floating dtype/device and are finite. T>=0, E,D>=1,
# 1<=k<=E. bias is [E] and only accepted for sigmoid_bias; otherwise
# it must be None. scale is positive finite for sigmoid_bias and must
# equal 1 in the other modes. Invalid inputs raise ValueError; do not
# mutate any input. Empty T returns correctly shaped empty tensors.

# %% [answer]
def route_variants(logits: torch.Tensor, expert_outputs: torch.Tensor,
                   k: int, mode: str = 'selected_softmax',
                   bias: torch.Tensor | None = None, scale: float = 1.0
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the routed mixture, selected IDs/weights, and full scores."""
    pass

# %%
# Baseline: the existing 0028 router normalizes only selected logits.
# With k=1 the selected weight is exactly one, independent of logit.
check('0095_router_variants_eval.py', stage='baseline')

# %%
# Follow-up: full_truncate loses the mass of unchosen experts. In
# full_renorm it is restored. Their selected weights differ, even
# though full_renorm and selected_softmax produce equal weights in exact
# arithmetic when the same experts are chosen. Why might exposing the
# full scores matter for an auxiliary load loss?
check('0095_router_variants_eval.py', stage='followup')

# %%
# Production: sigmoid_bias has selection-only bias and a route scale.
# Probe ties, non-contiguous tensors, empty T, invalid inputs, and
# gradients away from top-k ties. This exercises a public equation,
# not a claim about a checkpoint's private training implementation.
check('0095_router_variants_eval.py', stage='production')
