# %%
# mlp · MLP Modeling Lab
# Goal: Build a two-layer ReLU MLP in NumPy, then train, evaluate and checkpoint it on fixed 2-D splits.
# Each checkpoint is checked on its own, so they can pass in any order.
#
# Watch out: `REFERENCE_OVERRIDES = {"loss_and_grads"}` lets later checkpoints run on the reference component.
# Those passes are recorded as reference-assisted, never as independent credit.

# %%
# Init — `init_params(rng, input_dim, hidden_dim, classes)` returns `W1` $(D, H)$, `b1` $(H,)$, `W2` $(H, C)$ and `b2` $(C,)$, drawing only from the injected `rng`.
# Forward — `forward(X, params)` returns `hidden_pre`, `hidden` and `logits`, where `hidden` is $\max(0, h)$ of the pre-activation $h$.
# Backward — `loss_and_grads(X, y, params)` returns the mean softmax cross-entropy and a gradient for every parameter block.
# Watch out: the loss is a mean over the $B$ rows, so every gradient carries the $1/B$:
# $$\frac{\partial L}{\partial z} = \frac{p - \operatorname{onehot}(y)}{B}$$
# Train — `train_epoch` runs one shuffled mini-batch SGD epoch with momentum and returns new parameters, optimizer state and epoch statistics, leaving its inputs unchanged.
# Inference — `predict_proba(X, params, batch_size)` returns the same probabilities whatever the batch size.
# Evaluate — `evaluate_model` reports loss, accuracy and predictions without changing the parameters.
# State — `save_state` and `restore_state` round-trip the parameters and the optimizer's velocity as independent copies.
# Diagnose — A planted `loss_and_grads` trains badly. Compare it with directional finite differences, then return a diagnosis naming the defect, the failed invariant, the evidence, the repair and a regression check.
# Why: a central difference approximates the directional derivative,
# $$\frac{L(\theta + \epsilon v) - L(\theta - \epsilon v)}{2\epsilon} \approx \nabla L \cdot v$$
# so a gradient off by a constant factor shows up as a steady ratio, not as noise.

# %% [drill]
# Mean or sum?
# The loss averages 4 rows. One printed line is `db2`; predict both lines and say which.
import numpy as np
p = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.3, 0.3, 0.4], [0.2, 0.2, 0.6]])
y = np.array([0, 1, 2, 2])
dlogits = p.copy()
dlogits[np.arange(4), y] -= 1.0
print(dlogits.sum(axis=0))
print((dlogits / len(y)).sum(axis=0))

# %% [drill]
# Gradient through ReLU at zero
# Predict the gradient that reaches `hidden_pre` when the upstream gradient is all ones.
import numpy as np
hidden_pre = np.array([-1.0, 0.0, 2.0])
upstream = np.ones(3)
print(upstream * (hidden_pre > 0.0))

# %% [drill]
# Momentum's second step
# Two SGD steps with momentum 0.9 and learning rate 0.1 see the same gradient. Predict the velocity and the parameter after both.
velocity, param = 0.0, 1.0
for grad in [1.0, 1.0]:
    velocity = 0.9 * velocity + grad
    param = param - 0.1 * velocity
print(round(velocity, 4), round(param, 4))
