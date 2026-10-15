# %%
# 0083 · Learn from a replay tape
# Goal: Replay a recording of transitions through tabular Q-learning, in
# order, and return the updated $[S, A]$ Q table as a new array.
#
# Original dojo exercise inspired by Deep-ML's Q-learning topic:
# https://www.deep-ml.com/problems/133

# %%
import numpy as np
from torch_judge import check

# %%
# A robot records (state, action, reward, next_state, done) after every move.
# Q(s,a) estimates the return from choosing action a now. A recording is
# chronological: a later update sees earlier changes to the Q table.
# Predict: What is `Q[0,1]` after the first move below? What if `done` were True?
q_demo = np.array([[0., 0.], [4., 2.]])
target_demo = 2. + 0.5 * np.max(q_demo[1])
print('One-step target:', target_demo)

# %%
# Bellman update: the target is $y = r + \gamma \max_{a'} Q(s', a')$, unless
# `done`, when $y = r$. Then
# $$Q(s,a) \leftarrow Q(s,a) + \alpha\,\big(y - Q(s,a)\big)$$
# Watch out: use the UPDATED table for each next row of the recording. A loop
# over transitions is appropriate because the rows depend on earlier updates.

# %% [drill]
# Does the episode end here?
# With $\alpha = 1$ the new $Q(0, 0)$ equals the target. Predict it for reward $-2$, $\gamma = 0.9$ and next-state row $[100, 50]$, with `done` honored and ignored.
import numpy as np
q = np.array([[10., 2.], [100., 50.]])
reward, gamma = -2., 0.9
print("done honored:", reward, " done ignored:", reward + gamma * q[1].max())

# %% [drill]
# Old table or updated table?
# Replay the follow-up recording with $\alpha = \gamma = 0.5$. Predict column 0 when each target reads the live table, and when every target reads the table as it was before the replay.
import numpy as np
s, a, r, n = [0, 1, 0], [0, 0, 0], [1., 4., 2.], [1, 1, 1]
done = [False, True, False]
alpha, gamma = 0.5, 0.5
seq = np.zeros((2, 2))
for i in range(3):
    boot = 0. if done[i] else gamma * seq[n[i]].max()
    seq[s[i], a[i]] += alpha * (r[i] + boot - seq[s[i], a[i]])
frozen, batch = np.zeros((2, 2)), np.zeros((2, 2))
for i in range(3):
    boot = 0. if done[i] else gamma * frozen[n[i]].max()
    batch[s[i], a[i]] += alpha * (r[i] + boot - batch[s[i], a[i]])
print("sequential:", seq[:, 0], " targets from the old table:", batch[:, 0])

# %% [drill]
# Whose table did you update?
# Predict the caller's `q` after one update through `alias = q`, and after one through `q.copy()`.
import numpy as np
q = np.array([[1., 2.]])
alias = q
alias[0, 1] += 0.5 * (3. - alias[0, 1])
print("caller's q after an aliased update:", q)
q = np.array([[1., 2.]])
copy = q.copy()
copy[0, 1] += 0.5 * (3. - copy[0, 1])
print("caller's q after a copied update:", q)

# %% [answer]
def q_learning(q_values: np.ndarray, states: np.ndarray, actions: np.ndarray,
               rewards: np.ndarray, next_states: np.ndarray, dones: np.ndarray,
               alpha: float = 0.1, gamma: float = 0.9) -> np.ndarray:
    """Replay transitions in order and return a new [S,A] Q table."""
    pass

# %%
# Baseline — One transition and an empty recording. The empty case returns
# a separate copy, ready for another round of practice.
check('0083_q_learning_eval.py', stage='baseline')

# %%
# Follow-up — A recording revisits the same state-action pair. The middle
# transition ends an episode; it must not borrow value from its next state.
# Predict: Work the three updates on paper before running the verifier.
check('0083_q_learning_eval.py', stage='followup')

# %%
# Production — Contract: finite nonempty floating $[S, A]$ table, matching
# one-dimensional experience columns, integer in-range indices, boolean
# done flags, finite rewards, $0 < \alpha \le 1$ and $0 \le \gamma \le 1$.
# Reject invalid input with `ValueError`. Do not mutate the recording or Q.
check('0083_q_learning_eval.py', stage='production')
