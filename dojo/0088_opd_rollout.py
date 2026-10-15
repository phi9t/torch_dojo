# %%
# 0088 · The teacher follows the student's route
# Original dojo exercise about on-policy distillation:
# https://arxiv.org/abs/2306.13649
# https://arxiv.org/abs/2604.13016
# 0041_opd_loss.py first teaches reverse KL at already supplied positions.

# %%
import torch
import torch.nn.functional as F
from torch_judge import check

# %%
# A student and teacher both map integer prefixes [B,L] to next-token
# logits [B,V]. Only the STUDENT chooses the next token. The teacher is
# asked about the prefixes the student actually visits, including ones
# the teacher itself would never have chosen. Predict the second prefix
# when a student begins at [0] and deterministically samples token 1.
start = torch.tensor([[0]], dtype=torch.long)
print('first prefix:', start.tolist(), 'second prefix if token 1:',
      torch.cat((start, torch.ones((1, 1), dtype=torch.long)), dim=1).tolist())

# %%
# At each visited prefix, compute exact categorical reverse KL:
# sum_v p_student(v) * (log p_student(v) - log p_teacher(v)).
# Draw a token from the student's DETACHED distribution with an injected
# generator. Keep the student's logits in the KL graph; freeze the teacher.
# Return sampled tokens [B,steps] and the scalar mean of each row's mean
# valid-token KL. This is a tiny on-policy data-and-loss step, not a trainer.

# %% [answer]
def opd_rollout(student, teacher, start_tokens: torch.Tensor, steps: int,
                generator: torch.Generator,
                eos_token: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample student continuations and distill on their visited prefixes."""
    pass

# %%
# Baseline: two sampled steps, exact reverse KL, and a seeded replay.
check('0088_opd_rollout_eval.py', stage='baseline')

# %%
# Follow-up: the teacher prefers a different token; check that it still
# sees the student's next prefix. If EOS is sampled, fill remaining token
# slots with EOS and stop scoring that row.
check('0088_opd_rollout_eval.py', stage='followup')

# %%
# Production: require nonempty int64 prefixes, positive steps, a generator
# on the input device, and finite compatible [active,V] logits with V>=2.
# Validate prefix/EOS IDs, return ValueError for malformed inputs, leave
# the starting prefix unchanged, and allow variable EOS lengths in a batch.
# Teacher parameters and discrete draws receive no gradient; student
# parameters do. No policy update happens between sampling and scoring.
check('0088_opd_rollout_eval.py', stage='production')
