"""Reference and checks for tabular on-policy trajectory sampling."""

# %%
import numpy as np
from numbers import Integral, Real

# Drill 1: Interleaved prints `[(0, 0), (1, 1), (0, 0), (0, 0)]`; actions first
# prints `[(0, 0), (0, 0), (1, 0), (1, 0)]`. The generator yields the same
# stream of draws, `0 0 1 1 0 0 0 0`, either way; only the order of the calls
# decides which draw becomes an action and which a successor. A seed replays
# an episode exactly only if every implementation consumes the stream in the
# same order: action first, then successor, one step at a time, all from the
# injected `rng` rather than the global `np.random`.

# Drill 2: It prints `gamma**step: 4.0  gamma**(step+1): 2.0`. The first
# reward is earned now, so it carries $\gamma^0 = 1$: $G = 2 + 0.5 \cdot 4 = 4$.
# Starting the exponent at 1 discounts every reward one step too many and
# halves this return. Multiplying a running `discount` by $\gamma$ after
# adding each reward keeps the exponents right.

# Drill 3: It prints `[0.5 0.5 0. ]` and `[0. 1. 0.]`. Each episode really
# occupies room 0 before it moves, so counting the start state once per
# episode gives six visits split evenly. Counting only successors claims the
# robot never stood in room 0, and makes the raw visit counts stop adding up
# to `episodes + counts.sum()` before they are normalized.

# %%
def sample_trajectories(transitions, policy, rewards, start_state, terminal,
                        gamma, episodes, horizon, rng):
    try:
        transitions = np.asarray(transitions, dtype=np.float64)
        policy = np.asarray(policy, dtype=np.float64)
        rewards = np.asarray(rewards, dtype=np.float64)
        terminal = np.asarray(terminal)
    except (TypeError, ValueError) as error:
        raise ValueError('MDP arrays must be numeric') from error
    if transitions.ndim != 3:
        raise ValueError('transitions must have shape [S,A,S]')
    states, actions, successors = transitions.shape
    if (states < 1 or actions < 1 or successors != states
            or policy.shape != (states, actions)
            or rewards.shape != (states, actions, states)
            or terminal.shape != (states,) or terminal.dtype != np.bool_):
        raise ValueError('inconsistent MDP shapes or terminal mask')
    if (not np.isfinite(transitions).all() or not np.isfinite(policy).all()
            or not np.isfinite(rewards).all() or np.any(transitions < 0)
            or np.any(policy < 0)
            or not np.allclose(transitions.sum(axis=2), 1., atol=1e-8, rtol=0)
            or not np.allclose(policy.sum(axis=1), 1., atol=1e-8, rtol=0)):
        raise ValueError('transition and policy rows must be probabilities')
    if (not isinstance(start_state, Integral) or isinstance(start_state, (bool, np.bool_))
            or not 0 <= start_state < states
            or not isinstance(gamma, Real) or isinstance(gamma, (bool, np.bool_))
            or not np.isfinite(gamma) or not 0 <= gamma <= 1
            or not isinstance(episodes, Integral) or isinstance(episodes, (bool, np.bool_))
            or episodes < 1 or not isinstance(horizon, Integral)
            or isinstance(horizon, (bool, np.bool_)) or horizon < 1
            or not isinstance(rng, np.random.Generator)):
        raise ValueError('invalid start, discount, rollout size, or generator')

    paths = []
    visits = np.zeros(states, dtype=np.int64)
    counts = np.zeros((states, actions), dtype=np.int64)
    return_sum = 0.
    for _ in range(episodes):
        state = int(start_state)
        visits[state] += 1
        path = []
        discount = 1.
        episode_return = 0.
        for _ in range(horizon):
            if terminal[state]:
                break
            action = int(rng.choice(actions, p=policy[state] / policy[state].sum()))
            next_state = int(rng.choice(states, p=transitions[state, action]
                                        / transitions[state, action].sum()))
            reward = float(rewards[state, action, next_state])
            done = bool(terminal[next_state])
            path.append((state, action, next_state, reward, done))
            counts[state, action] += 1
            visits[next_state] += 1
            episode_return += discount * reward
            discount *= gamma
            state = next_state
            if done:
                break
        paths.append(path)
        return_sum += episode_return
    return paths, visits / visits.sum(), counts, return_sum / episodes


# %% [verifiers]
def _chain():
    # 0 -> 1 -> 2, with terminal state 2. Terminal rows remain valid tables.
    transitions = np.array([[[0., 1., 0.]],
                            [[0., 0., 1.]],
                            [[0., 0., 1.]]])
    rewards = np.zeros((3, 1, 3))
    rewards[0, 0, 1] = 2.
    rewards[1, 0, 2] = 4.
    return transitions, np.ones((3, 1)), rewards, np.array([False, False, True])


def verify_baseline_hand_trace(sample_trajectories):
    p, pi, r, terminal = _chain()
    paths, visits, counts, mean_return = sample_trajectories(
        p, pi, r, 0, terminal, 0.5, 2, 5, np.random.default_rng(5))
    expected = [(0, 0, 1, 2., False), (1, 0, 2, 4., True)]
    assert paths == [expected, expected]
    assert np.allclose(visits, [1/3, 1/3, 1/3])
    assert np.array_equal(counts, [[2], [2], [0]])
    assert np.isclose(mean_return, 4.)


def verify_baseline_horizon_counts_initial_and_successors(sample_trajectories):
    p, pi, r, terminal = _chain()
    paths, visits, counts, mean_return = sample_trajectories(
        p, pi, r, 0, terminal, 0.5, 3, 1, np.random.default_rng(2))
    assert paths == [[(0, 0, 1, 2., False)]] * 3
    assert np.allclose(visits, [0.5, 0.5, 0.])
    assert np.array_equal(counts, [[3], [0], [0]])
    assert np.isclose(mean_return, 2.)


def verify_followup_samples_policy_and_transition_with_injected_rng(sample_trajectories):
    p = np.array([[[.2, .8], [.75, .25]],
                  [[.4, .6], [.1, .9]]])
    pi = np.array([[.35, .65], [.8, .2]])
    r = np.array([[[1., 3.], [5., 7.]],
                  [[11., 13.], [17., 19.]]])
    terminal = np.array([False, True])
    seed = 31
    # Independent hand-sized simulation using the public sampling order.
    expected_paths = []
    expected_counts = np.zeros((2, 2), dtype=int)
    expected_visits = np.zeros(2, dtype=int)
    expected_returns = []
    replay = np.random.default_rng(seed)
    for _ in range(12):
        state = 0
        expected_visits[state] += 1
        path = []
        total = 0.
        for t in range(4):
            action = int(replay.choice(2, p=pi[state]))
            next_state = int(replay.choice(2, p=p[state, action]))
            reward = float(r[state, action, next_state])
            done = bool(terminal[next_state])
            path.append((state, action, next_state, reward, done))
            expected_counts[state, action] += 1
            expected_visits[next_state] += 1
            total += 0.7**t * reward
            state = next_state
            if done:
                break
        expected_paths.append(path)
        expected_returns.append(total)
    actual = sample_trajectories(p, pi, r, 0, terminal, 0.7, 12, 4,
                                 np.random.default_rng(seed))
    paths, visits, counts, mean_return = actual
    assert paths == expected_paths
    assert np.allclose(visits, expected_visits / expected_visits.sum())
    assert np.array_equal(counts, expected_counts)
    assert np.isclose(mean_return, np.mean(expected_returns))
    assert np.unique(np.array([row[1] for path in paths for row in path])).size == 2


def verify_followup_seed_replay_and_changed_seed(sample_trajectories):
    p = np.full((2, 2, 2), 0.5)
    pi = np.full((2, 2), 0.5)
    r = np.arange(8., dtype=float).reshape(2, 2, 2)
    args = (p, pi, r, 0, np.zeros(2, dtype=bool), .9, 20, 8)
    a = sample_trajectories(*args, np.random.default_rng(11))
    b = sample_trajectories(*args, np.random.default_rng(11))
    c = sample_trajectories(*args, np.random.default_rng(12))
    assert a[0] == b[0] and all(np.array_equal(x, y) for x, y in zip(a[1:3], b[1:3]))
    assert a[3] == b[3]
    assert a[0] != c[0]


def verify_production_terminal_start_has_no_actions(sample_trajectories):
    p, pi, r, terminal = _chain()
    result = sample_trajectories(p, pi, r, 2, terminal, .5, 3, 5,
                                 np.random.default_rng(1))
    assert result[0] == [[], [], []]
    assert np.array_equal(result[1], [0., 0., 1.])
    assert np.array_equal(result[2], np.zeros((3, 1)))
    assert result[3] == 0.


def verify_production_count_conservation_and_no_mutation(sample_trajectories):
    p = np.full((3, 2, 3), 1/3)
    pi = np.full((3, 2), .5)
    r = np.arange(18., dtype=float).reshape(3, 2, 3)
    terminal = np.array([False, False, True])
    copies = [x.copy() for x in (p, pi, r, terminal)]
    paths, visits, counts, mean_return = sample_trajectories(
        p, pi, r, 0, terminal, .8, 30, 6, np.random.default_rng(21))
    observed_counts = np.zeros((3, 2), dtype=int)
    observed_visits = np.zeros(3, dtype=int)
    for path in paths:
        observed_visits[0] += 1
        assert len(path) <= 6
        for i, (state, action, next_state, reward, done) in enumerate(path):
            observed_counts[state, action] += 1
            observed_visits[next_state] += 1
            assert reward == r[state, action, next_state]
            assert done == bool(terminal[next_state])
            if i:
                assert path[i-1][2] == state and not path[i-1][4]
        if path:
            assert path[-1][4] or len(path) == 6
    assert np.array_equal(counts, observed_counts)
    assert np.allclose(visits, observed_visits / observed_visits.sum())
    assert int(observed_visits.sum()) == 30 + int(counts.sum())
    assert np.isfinite(mean_return)
    assert all(np.array_equal(x, y) for x, y in zip((p, pi, r, terminal), copies))


def verify_production_rejects_invalid_contract(sample_trajectories):
    p, pi, r, terminal = _chain()
    valid = (p, pi, r, 0, terminal, .9, 2, 3, np.random.default_rng(1))
    bad = [
        (np.ones((3, 1, 2)), *valid[1:]),
        (p, np.ones((2, 1)), *valid[2:]),
        (p, pi, np.ones((3, 1)), *valid[3:]),
        (p, pi, r, 3, *valid[4:]),
        (p, pi, r, True, *valid[4:]),
        (p, pi, r, 0, np.array([0, 0, 1]), *valid[5:]),
        (p, pi, r, 0, terminal, -0.1, *valid[6:]),
        (p, pi, r, 0, terminal, 1.1, *valid[6:]),
        (p, pi, r, 0, terminal, .9, 0, *valid[7:]),
        (p, pi, r, 0, terminal, .9, 2, 0, valid[8]),
        (*valid[:8], None),
        (np.where(p == 1, .8, p), *valid[1:]),
        (np.where(p == 1, np.nan, p), *valid[1:]),
        (p, np.full((3, 1), .9), *valid[2:]),
        (p, pi, np.full((3, 1, 3), np.inf), *valid[3:]),
    ]
    for args in bad:
        try:
            sample_trajectories(*args)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid trajectory input accepted')


# %%
from torch_judge import evaluate
if not evaluate('0091_on_policy_trajectories_eval.py', sample_trajectories):
    raise SystemExit(1)
