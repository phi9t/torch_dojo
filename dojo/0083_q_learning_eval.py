"""Reference and checks for sequential tabular Q-learning updates."""

# %%
import numpy as np
from numbers import Real
from hypothesis import given, settings, strategies as st

# Drill 1: It prints `done honored: -2.0  done ignored: 88.0`. A done flag
# means $s'$ is past the end of the episode, so there is no future return to
# bootstrap: $y = r = -2$. Ignoring it adds $0.9 \cdot \max(100, 50) = 90$ of
# value the robot can never collect, and that phantom value then spreads to
# every state that leads here.

# Drill 2: Sequential prints `[1.75 2.  ]`; old-table targets print
# `[1.25 2.  ]`. The second move sets $Q(1,0) = 2$ before the third move
# bootstraps from row 1, so the third target is $2 + 0.5 \cdot 2 = 3$ and
# $Q(0,0) = 0.5 + 0.5\,(3 - 0.5) = 1.75$. Reading the frozen table sees
# $\max Q(1,\cdot) = 0$, a target of 2 and only 1.25. A replay tape is
# chronological, which is why the reference loops over rows.

# Drill 3: The aliased update prints `[[1.  2.5]]`: `alias = q` names the same
# array, so the update rewrites the caller's table. The copied update leaves
# `[[1. 2.]]`. Start from `q_values.copy()` (after converting to float), which
# is also why an empty recording must return a separate array.

# %%
def q_learning(q_values, states, actions, rewards, next_states, dones,
               alpha=0.1, gamma=0.9):
    q_values = np.asarray(q_values)
    states = np.asarray(states)
    actions = np.asarray(actions)
    rewards = np.asarray(rewards, dtype=np.float64)
    next_states = np.asarray(next_states)
    dones = np.asarray(dones)
    if (q_values.ndim != 2 or min(q_values.shape) < 1
            or q_values.dtype.kind != 'f'
            or not np.isfinite(q_values).all()):
        raise ValueError('q_values must be a nonempty finite [S,A] table')
    q_values = q_values.astype(np.float64)
    count = len(states) if states.ndim == 1 else -1
    if (count < 0 or actions.shape != (count,) or rewards.shape != (count,)
            or next_states.shape != (count,) or dones.shape != (count,)
            or (count > 0 and (states.dtype.kind not in 'iu'
                               or actions.dtype.kind not in 'iu'
                               or next_states.dtype.kind not in 'iu'
                               or dones.dtype != np.bool_))
            or not np.isfinite(rewards).all()):
        raise ValueError('experience columns need matching one-dimensional types')
    if (np.any(states < 0) or np.any(states >= q_values.shape[0])
            or np.any(next_states < 0) or np.any(next_states >= q_values.shape[0])
            or np.any(actions < 0) or np.any(actions >= q_values.shape[1])):
        raise ValueError('experience index outside Q table')
    if (not isinstance(alpha, Real) or isinstance(alpha, (bool, np.bool_))
            or not np.isfinite(alpha) or not 0 < alpha <= 1
            or not isinstance(gamma, Real) or isinstance(gamma, (bool, np.bool_))
            or not np.isfinite(gamma) or not 0 <= gamma <= 1):
        raise ValueError('alpha must be in (0,1] and gamma in [0,1]')
    updated = q_values.copy()
    for state, action, reward, next_state, done in zip(
            states, actions, rewards, next_states, dones):
        bootstrap = 0. if done else gamma * np.max(updated[next_state])
        current = updated[state, action]
        updated[state, action] = current + alpha * (reward + bootstrap - current)
        if not np.isfinite(updated[state, action]):
            raise ValueError('Q update overflowed')
    return updated


# %% [verifiers]
def verify_baseline_one_bellman_update(q_learning):
    q = np.array([[0., 0.], [4., 2.]])
    actual = q_learning(q, [0], [1], [2.], [1], [False], alpha=0.25, gamma=0.5)
    assert np.allclose(actual, [[0., 1.], [4., 2.]])


def verify_baseline_empty_replay_returns_copy(q_learning):
    q = np.array([[1., 2.]])
    actual = q_learning(q, [], [], [], [], [])
    assert np.array_equal(actual, q) and actual is not q


def verify_followup_updates_in_recorded_order(q_learning):
    q = np.zeros((2, 2))
    result = q_learning(q, [0, 1, 0], [0, 0, 0], [1., 4., 2.],
                        [1, 1, 1], [False, True, False], alpha=0.5, gamma=0.5)
    assert np.allclose(result, [[1.75, 0.], [2., 0.]])


def verify_followup_terminal_discards_bootstrap(q_learning):
    q = np.array([[10., 2.], [100., 50.]])
    result = q_learning(q, [0], [0], [-2.], [1], [True], alpha=1., gamma=0.9)
    assert result[0, 0] == -2. and result[1, 0] == 100.


def verify_production_rejects_invalid_inputs(q_learning):
    q = np.zeros((2, 2))
    valid = (q, [0], [0], [1.], [1], [False])
    invalid = [
        (*valid, 0., 0.9), (*valid, 1.1, 0.9),
        (*valid, 0.1, -0.1), (*valid, 0.1, 1.1),
        (*valid, 'bad', 0.9), (*valid, 0.1, 'bad'),
        (q, [2], [0], [1.], [1], [False], 0.1, 0.9),
        (q, [0], [2], [1.], [1], [False], 0.1, 0.9),
        (q, [0], [0], [np.nan], [1], [False], 0.1, 0.9),
        (q, [0], [0], [1.], [1], [0], 0.1, 0.9),
        (q, [0, 1], [0], [1.], [1], [False], 0.1, 0.9),
        (np.zeros((0, 2)), [], [], [], [], [], 0.1, 0.9),
        (np.zeros((2, 2), dtype=np.int64), [], [], [], [], [], 0.1, 0.9),
    ]
    for args in invalid:
        try:
            q_learning(*args)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid replay accepted')


def verify_production_inputs_unchanged(q_learning):
    q = np.array([[1., 2.], [3., 4.]])
    states = np.array([0, 1]); rewards = np.array([5., 6.])
    q0, s0, r0 = q.copy(), states.copy(), rewards.copy()
    q_learning(q, states, [0, 1], rewards, [1, 0], [False, True])
    assert np.array_equal(q, q0) and np.array_equal(states, s0) and np.array_equal(rewards, r0)


def verify_production_generated_terminal_update(q_learning):
    @settings(max_examples=20, deadline=None, derandomize=True)
    @given(st.floats(-10, 10, allow_nan=False, allow_infinity=False),
           st.floats(-10, 10, allow_nan=False, allow_infinity=False))
    def check(old, reward):
        q = np.array([[old, 7.], [100., 200.]])
        actual = q_learning(q, [0], [0], [reward], [1], [True], alpha=0.2, gamma=0.9)
        assert np.isclose(actual[0, 0], old + 0.2 * (reward - old))
    check()


# %%
from torch_judge import evaluate
if not evaluate('0083_q_learning_eval.py', q_learning):
    raise SystemExit(1)
