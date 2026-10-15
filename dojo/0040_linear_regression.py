# %%
# 0040 · Linear regression, three routes to the same model
# Work through the cells in Emacs/Jupyter. Keep all three methods inside the
# one editable answer class. X is [N,D], y is [N], prediction X@w+b.

# %%
import torch
from torch import nn

# %%
# Baseline: return w [D] and scalar b from each method. The closed form uses
# an augmented design matrix and least squares. Manual GD starts at zero,
# uses MSE gradients, and must not invoke autograd. nn_linear uses nn.Linear
# and backward. Predict the augmented column's role before running this.
X_probe = torch.tensor([[0., 1.], [1., 2.], [2., 3.]])
print('augmented design:', torch.cat([X_probe, torch.ones(3, 1)], dim=1))

# %% [answer]
class LinearRegression:
    def closed_form(self, X: torch.Tensor, y: torch.Tensor):
        pass

    def gradient_descent(self, X: torch.Tensor, y: torch.Tensor,
                         lr: float = 0.01, steps: int = 1000):
        pass

    def nn_linear(self, X: torch.Tensor, y: torch.Tensor,
                  lr: float = 0.01, steps: int = 1000):
        pass

# %%
from torch_judge import check
check('0040_linear_regression_eval.py', stage='baseline')

# %%
# Follow-up: rank-deficient design needs a least-squares solution. For two
# duplicate columns the minimum-norm solution splits their combined weight
# evenly. Preserve input dtype/device and do not mutate X or y.
check('0040_linear_regression_eval.py', stage='followup')

# %%
# Production: ill conditioning makes inversion of X.T@X fragile because
# it squares the condition number. Prefer torch.linalg.lstsq. Reject empty,
# nonfinite, or incompatible data; training requires positive finite lr and
# nonnegative integer steps. At zero GD steps, return the zero initializer.
check('0040_linear_regression_eval.py', stage='production')
