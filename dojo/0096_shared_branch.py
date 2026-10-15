# %%
# 0096 · The expert everyone sees
# 0095 chooses sparse experts. A shared expert runs for EVERY token.
# Sources: DeepSeek-V2 Eq. 20 https://arxiv.org/html/2405.04434#S2.SS2.SSS1
# DeepSeek-V3 Eq. 12 https://arxiv.org/html/2412.19437#S2.SS1.SSS2
# Qwen3 report (no shared branch) https://arxiv.org/html/2505.09388#S2
# Qwen3-Next compatible forward (sigmoid gate):
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_next/modeling_qwen3_next.py
# The last link is a compatible forward, not the private training loop.

# %%
import torch
from torch import nn
from torch_judge import check

# %%
# Suppose a routed expert gives 2 and a shared expert gives 3. Which
# output belongs to absent, ungated, and a gate of sigmoid(0)=1/2?
print('three possibilities:', 2., 2. + 3., 2. + .5 * 3.)
# Write a composer around a supplied shared nn.Module. A real shared
# expert can be an FFN, but a small Linear makes the behavior visible.
# forward(x, routed) combines tensors of the same [...,D] shape, where
# rank>=2, D>=1 and leading dimensions may contain zero. Inputs must
# be finite, floating, same dtype/device. Invalid inputs raise ValueError.
#
# mode='absent': shared_expert and gate are None; return routed.
# mode='ungated': shared_expert is an nn.Module mapping [...,D] to
# [...,D]; return routed + shared_expert(x).
# mode='gated': additionally require a gate nn.Module mapping [...,D]
# to [...,1]. Return routed + sigmoid(gate(x))*shared_expert(x).
# Reject incompatible modules at construction and wrong/nonfinite
# module outputs at call time. Register supplied modules so their
# parameters receive gradients. Do not normalize the shared branch
# together with routed weights, and do not mutate input tensors.

# %% [answer]
class SharedBranch(nn.Module):
    def __init__(self, mode: str, shared_expert: nn.Module | None = None,
                 gate: nn.Module | None = None):
        super().__init__()
        pass

    def forward(self, x: torch.Tensor, routed: torch.Tensor) -> torch.Tensor:
        pass

# %%
# Baseline: absent and ungated agree with hand-computed sums. An
# ungated shared path still contributes when routed output is zero.
check('0096_shared_branch_eval.py', stage='baseline')

# %%
# Follow-up: gate output is a LOGIT. Apply sigmoid to it, not to the
# shared expert's result. Probe two tokens with different gates and
# check that the ungated branch can change independently of routed.
check('0096_shared_branch_eval.py', stage='followup')

# %%
# Production: preserve autograd through routed, shared, gate and x;
# handle strided/empty tensors and reject invalid module outputs.
# A token-dependent gate prevents collapsing this branch into a single
# ungated FFN by merely concatenating expert weights.
check('0096_shared_branch_eval.py', stage='production')
