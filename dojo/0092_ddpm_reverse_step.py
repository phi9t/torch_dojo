# %%
# 0092 · One DDPM reverse step
# Original dojo exercise prompted by https://www.deep-ml.com/problems/396

# %%
import torch
from torch_judge import check

# %%
# With betas [0.1, 0.2], x_1 = 2, and predicted noise = 1, the first
# reverse step has mean (2 - 0.1 / sqrt(0.1)) / sqrt(0.9).
# Predict its value and whether sampling consumes any random numbers.
print('one-based step 1 mean:', (2 - .1 / .1**.5) / .9**.5)

# %%
# Inputs: x_t and predicted_noise are finite floating tensors of the same
# nonempty shape, dtype, and device. betas is a finite floating [T] tensor
# on that dtype/device, with every beta strictly between 0 and 1.
# step is an integer in 1..T, shared by every element of x_t.
# alpha_t = 1 - beta_t; alpha_bar_t = product(alpha_1..alpha_t).
# Return the noise-prediction posterior sample:
# mean = (x_t - beta_t * predicted_noise / sqrt(1-alpha_bar_t)) / sqrt(alpha_t)
# variance = beta_t * (1-alpha_bar_(t-1)) / (1-alpha_bar_t),
# where alpha_bar_0 = 1. At step 1 return only mean: do not draw noise
# or advance generator, even if one is supplied.

# %% [answer]
def ddpm_reverse_step(x_t: torch.Tensor, predicted_noise: torch.Tensor,
                      betas: torch.Tensor, step: int,
                      noise: torch.Tensor | None = None,
                      generator: torch.Generator | None = None) -> torch.Tensor:
    """Sample one reverse step using supplied noise or an injected generator."""
    pass

# %%
# Baseline: calculate the one-based posterior mean by hand.
check('0092_ddpm_reverse_step_eval.py', stage='baseline')

# %%
# Follow-up: step 1 has zero stochastic term. Earlier reverse steps add
# sqrt(posterior variance) times supplied or seeded standard-normal noise.
check('0092_ddpm_reverse_step_eval.py', stage='followup')

# %%
# Production: reject invalid step, schedule, shape, dtype, device, and
# nonfinite values with ValueError. Optional noise must match x_t. Preserve
# inputs and gradients through x_t and predicted_noise. When noise is absent
# at step > 1, draw on x_t's device and dtype from generator if provided.
check('0092_ddpm_reverse_step_eval.py', stage='production')
