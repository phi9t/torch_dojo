# %%
# 0091 · Follow a policy through a tiny world
# Goal: Sample episodes from a tabular MDP by following a stochastic policy
# with an injected, seeded generator, and return the paths, visit
# frequencies, state-action counts and mean discounted return.
#
# Original dojo exercise inspired by Deep-ML's on-policy sampling topic:
# https://www.deep-ml.com/problems/529
# 0082_policy_evaluation.py computes expected values without sampling;
# here, one policy generates actual episodes and empirical statistics.

# %%
import numpy as np
from torch_judge import check

# %%
# A robot starts in room 0. Its only action moves it to room 1 for reward 2,
# then to terminal room 2 for reward 4.
# Predict: What are its transition records, state-visit frequencies, and
# return when $\gamma = 0.5$? Is room 2 visited?
demo_transitions = np.array([[[0., 1., 0.]],
                             [[0., 0., 1.]],
                             [[0., 0., 1.]]])
demo_policy = np.ones((3, 1))
demo_rewards = np.zeros((3, 1, 3))
demo_rewards[0, 0, 1], demo_rewards[1, 0, 2] = 2., 4.
demo_terminal = np.array([False, False, True])
print('discounted return:', 2. + 0.5 * 4.)

# %%
# Inputs: transitions[S,A,S] gives Pr(next state | state, action),
# policy[S,A] gives Pr(action | state), and rewards[S,A,S] gives the reward
# for the sampled transition. terminal[S] is boolean. S,A >= 1.
# Each row of transitions and policy is finite, nonnegative, and sums to 1
# within absolute tolerance 1e-8. Rewards are finite. start_state is an
# integer in $[0, S)$; $\gamma$ is finite in $[0, 1]$; episodes and horizon are
# positive integers; `rng` is an injected `np.random.Generator`.
# Raise `ValueError` for invalid inputs. Do not mutate any input array.
#
# Return (paths, visits, counts, mean_return): paths is a list of ordered
# episodes, each a list of (state, action, next_state, reward, terminal)
# records. visits[S] is the fraction of ALL counted state visits, and
# counts[S,A] is the integer number of sampled state-action pairs.
# Count the start state once per episode and each reached successor once.
#
# Watch out: never choose an action from a terminal state. A terminal start
# gives an empty path, one visit, and zero return. A horizon-ended path keeps
# the last record's terminal flag False.
#
# Discount each episode's rewards by $\gamma^t$ (the first reward has $t = 0$),
# $$G = \sum_{t=0}^{T-1} \gamma^t r_t,$$
# then average $G$ by episode count. Sample the action first and the successor
# second, using `rng.choice` on their rows.

# %% [drill]
# Same seed, different episodes
# Both loops draw eight times from `default_rng(3)`. Predict the (action, successor) pairs when draws alternate, and when all actions are drawn first.
import numpy as np
pi = np.array([.5, .5]); p = np.array([.5, .5])
rng = np.random.default_rng(3)
interleaved = [(int(rng.choice(2, p=pi)), int(rng.choice(2, p=p))) for _ in range(4)]
rng = np.random.default_rng(3)
acts = [int(rng.choice(2, p=pi)) for _ in range(4)]
succ = [int(rng.choice(2, p=p)) for _ in range(4)]
print("action, successor interleaved:", interleaved)
print("all actions, then successors: ", list(zip(acts, succ)))

# %% [drill]
# Which step is undiscounted?
# Rewards 2 then 4 with $\gamma = 0.5$. Predict the return with exponents starting at 0 and starting at 1.
import numpy as np
rewards, gamma = np.array([2., 4.]), 0.5
steps = np.arange(len(rewards))
print("gamma**step:", (gamma ** steps * rewards).sum(), " gamma**(step+1):", (gamma ** (steps + 1) * rewards).sum())

# %% [drill]
# Does the start state count?
# Three horizon-1 episodes each record one move $0 \to 1$. Predict the visit frequencies with and without counting the start state.
import numpy as np
successors = np.array([1, 1, 1])
with_start = np.bincount(np.r_[[0, 0, 0], successors], minlength=3)
only_successors = np.bincount(successors, minlength=3)
print(with_start / with_start.sum(), only_successors / only_successors.sum())

# %% [answer]
def sample_trajectories(transitions: np.ndarray, policy: np.ndarray,
                        rewards: np.ndarray, start_state: int,
                        terminal: np.ndarray, gamma: float,
                        episodes: int, horizon: int,
                        rng: np.random.Generator):
    """Sample episodes and return paths, visit frequencies, counts, mean return."""
    pass

# %%
# Baseline — Deterministic two-step and horizon-truncated episodes.
# Check every counted visit: initial state plus each reached successor.
check('0091_on_policy_trajectories_eval.py', stage='baseline')

# %%
# Follow-up — Both the policy and the world are stochastic. Replay a seed
# exactly, and stop an episode when it reaches a terminal state.
check('0091_on_policy_trajectories_eval.py', stage='followup')

# %%
# Production — Validate every row and shape, handle a terminal start, and
# conserve visits and state-action counts across all transition records.
check('0091_on_policy_trajectories_eval.py', stage='production')
