# %%
# 0082 · How much is a policy worth?
# Goal: Return the value $V(s)$ of every state of a tabular MDP under a fixed
# policy, by iterating the Bellman expectation equation until it converges.
#
# Original dojo exercise inspired by Deep-ML's gridworld policy evaluation:
# https://www.deep-ml.com/problems/142

# %%
import numpy as np
from torch_judge import check

# %%
# Imagine a robot in a two-room maze. In room 0, its only action earns 3
# points and moves to room 1. Room 1 ends the episode.
# Predict: What are $V(0)$ and $V(1)$? Does room 1's tempting 99-point reward count?
transitions = np.array([[[0., 1.]], [[0., 1.]]])  # [state, action, next state]
rewards = np.array([[3.], [99.]])                  # [state, action]
policy = np.ones((2, 1))                          # Pr(action | state)
terminal = np.array([False, True])
print('First action rewards:', rewards[:, 0])

# %%
# For a nonterminal state, apply the Bellman expectation equation:
# $$V_{k+1}(s) = \sum_a \pi(a \mid s) \Big(R(s,a) + \gamma \sum_{s'} P(s' \mid s,a)\, V_k(s')\Big)$$
# Watch out: a terminal state's value is zero, regardless of its reward and
# transition rows.
#
# Work synchronously: compute every new value from the previous whole vector
# $V_k$. Stop when the largest absolute change is at most `tolerance`.

# %% [drill]
# The tempting terminal reward
# Predict both value vectors: one sweep loop ignores the terminal mask, the other zeroes terminal states after every sweep.
import numpy as np
P = np.array([[[0., 1.]], [[0., 1.]]]); R = np.array([[3.], [99.]])
terminal = np.array([False, True])
for mask in (False, True):
    v = np.zeros(2)
    for _ in range(2000):
        v = R[:, 0] + 0.9 * P[:, 0] @ v
        if mask:
            v[terminal] = 0
    print("masked" if mask else "unmasked", np.round(v, 6))

# %% [drill]
# Synchronous or in place?
# On the chain $0 \to 1 \to 2$ (terminal) with reward 1 per move, predict the values after ONE sweep from zeros, synchronously and in place in the order 2, 1, 0.
import numpy as np
nxt = np.array([1, 2, 2]); r = np.array([1., 1., 0.]); terminal = np.array([False, False, True])
sync = np.zeros(3); old = sync.copy()
sync = np.where(terminal, 0., r + 0.9 * old[nxt])
inplace = np.zeros(3)
for s in (2, 1, 0):
    inplace[s] = 0. if terminal[s] else r[s] + 0.9 * inplace[nxt[s]]
print("synchronous:", sync, " in-place 2,1,0:", inplace)

# %% [drill]
# How far does the discount reach?
# A self-loop earns 1 every step. Predict its value and how many sweeps reach tolerance $10^{-10}$ for $\gamma = 0.5, 0.9, 0.99$.
import numpy as np
for gamma in (0.5, 0.9, 0.99):
    v, sweeps = 0., 0
    while True:
        new = 1. + gamma * v
        sweeps += 1
        if abs(new - v) <= 1e-10:
            break
        v = new
    print(gamma, round(new, 6), sweeps)

# %% [answer]
def evaluate_policy(transitions: np.ndarray, rewards: np.ndarray,
                    policy: np.ndarray, terminal: np.ndarray,
                    gamma: float = 0.9, tolerance: float = 1e-10,
                    max_iterations: int = 10000) -> np.ndarray:
    """Return values [S] for a fixed tabular policy; do not mutate inputs."""
    pass

# %%
# Baseline — One action per state. Check the terminal room and a discounted
# self-loop that earns 1 point on every step.
# Predict: What is the self-loop's value at $\gamma = 0.5$?
check('0082_policy_evaluation_eval.py', stage='baseline')

# %%
# Follow-up — The robot can choose between two actions with different odds.
# Your [S,A,S] transition rows and [S,A] policy rows are probabilities.
# Compare the result with a direct linear-system solution on a tiny maze.
check('0082_policy_evaluation_eval.py', stage='followup')

# %%
# Production — Contract: require $S, A \ge 1$; finite rewards and probability
# rows; boolean terminal mask; $0 \le \gamma < 1$, positive tolerance and
# iteration limit. Raise `ValueError` for bad input and `RuntimeError` when the
# limit expires before convergence. Keep the input arrays unchanged.
check('0082_policy_evaluation_eval.py', stage='production')
