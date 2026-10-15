"""Reference and checks for an authored tabular policy-evaluation exercise."""

# %%
import numpy as np
from numbers import Real
from hypothesis import given, settings, strategies as st

# Drill 1: Unmasked prints `[894. 990.]`; masked prints `[3. 0.]`. Without the
# mask, room 1 bootstraps from itself: $V(1) = 99 + 0.9\,V(1)$ converges to 990,
# and room 0 inherits $3 + 0.9 \cdot 990 = 894$. A terminal state ends the
# episode, so nothing after it, including its own reward row, may count;
# zeroing it after every sweep keeps $V(1) = 0$ and $V(0) = 3$.

# Drill 2: Synchronous prints `[1. 1. 0.]`; in place in order 2, 1, 0 prints
# `[1.9 1. 0.]`, because state 0 already saw state 1's fresh value inside the
# same sweep. Both reach the same fixed point, but the sweep-by-sweep values
# and the change measured against `tolerance` differ, so this workout asks
# for $V_{k+1}$ computed entirely from $V_k$.

# Drill 3: It prints `0.5 2.0 35`, `0.9 10.0 220` and `0.99 100.0 2293`. The
# value is the geometric series $\sum_t \gamma^t = 1/(1-\gamma)$, and the error
# shrinks by a factor $\gamma$ per sweep, so the sweep count grows roughly like
# $1/(1-\gamma)$. That is why $\gamma$ must stay below 1 and why a
# `max_iterations` cap needs a loud `RuntimeError`.

# %%
def evaluate_policy(transitions, rewards, policy, terminal, gamma=0.9,
                    tolerance=1e-10, max_iterations=10000):
    transitions = np.asarray(transitions, dtype=np.float64)
    rewards = np.asarray(rewards, dtype=np.float64)
    policy = np.asarray(policy, dtype=np.float64)
    terminal = np.asarray(terminal)
    if transitions.ndim != 3:
        raise ValueError('transitions must have shape [S,A,S]')
    states, actions, next_states = transitions.shape
    if (states < 1 or actions < 1 or next_states != states
            or rewards.shape != (states, actions)
            or policy.shape != (states, actions)
            or terminal.shape != (states,) or terminal.dtype != np.bool_):
        raise ValueError('inconsistent MDP shapes or terminal mask')
    if (not np.isfinite(transitions).all() or not np.isfinite(rewards).all()
            or not np.isfinite(policy).all() or np.any(transitions < 0)
            or np.any(policy < 0)
            or not np.allclose(transitions.sum(axis=-1), 1, atol=1e-8, rtol=0)
            or not np.allclose(policy.sum(axis=-1), 1, atol=1e-8, rtol=0)):
        raise ValueError('transitions and policy must be probability rows')
    if (not isinstance(gamma, Real) or isinstance(gamma, (bool, np.bool_))
            or not np.isfinite(gamma) or not 0 <= gamma < 1
            or not isinstance(tolerance, Real) or isinstance(tolerance, (bool, np.bool_))
            or not np.isfinite(tolerance)
            or tolerance <= 0 or isinstance(max_iterations, (bool, np.bool_))
            or not isinstance(max_iterations, (int, np.integer))
            or max_iterations < 1):
        raise ValueError('invalid iteration settings')
    values = np.zeros(states, dtype=np.float64)
    for _ in range(max_iterations):
        action_values = rewards + gamma * np.einsum('san,n->sa', transitions, values)
        updated = np.sum(policy * action_values, axis=1)
        updated[terminal] = 0
        if not np.isfinite(updated).all():
            raise ValueError('value iteration overflowed')
        if np.max(np.abs(updated - values)) <= tolerance:
            return updated
        values = updated
    raise RuntimeError('policy evaluation did not converge within max_iterations')


# %% [verifiers]
def verify_baseline_one_step_and_absorbing_state(evaluate_policy):
    transitions = np.array([[[0., 1.]], [[0., 1.]]])
    rewards = np.array([[3.], [99.]])
    policy = np.ones((2, 1))
    terminal = np.array([False, True])
    result = evaluate_policy(transitions, rewards, policy, terminal)
    assert result.shape == (2,)
    assert np.allclose(result, [3., 0.], atol=1e-9)


def verify_baseline_discounted_loop(evaluate_policy):
    result = evaluate_policy(np.ones((1, 1, 1)), np.array([[1.]]),
                             np.ones((1, 1)), np.array([False]), gamma=0.5)
    assert np.allclose(result, [2.], atol=1e-8)


def verify_followup_stochastic_policy_against_linear_solve(evaluate_policy):
    transitions = np.array([
        [[0.5, 0.5, 0.], [0., 0.25, 0.75]],
        [[0., 0.5, 0.5], [1., 0., 0.]],
        [[0., 0., 1.], [0., 0., 1.]],
    ])
    rewards = np.array([[2., -1.], [1., 3.], [99., 99.]])
    policy = np.array([[0.25, 0.75], [0.8, 0.2], [0.5, 0.5]])
    terminal = np.array([False, False, True])
    p_pi = np.einsum('sa,san->sn', policy, transitions)
    r_pi = (policy * rewards).sum(axis=1)
    p_pi[terminal] = 0
    r_pi[terminal] = 0
    expected = np.linalg.solve(np.eye(3) - 0.8 * p_pi, r_pi)
    actual = evaluate_policy(transitions, rewards, policy, terminal, gamma=0.8)
    assert np.allclose(actual, expected, atol=1e-8)


def verify_followup_gamma_zero_and_negative_reward(evaluate_policy):
    p = np.array([[[1., 0.]], [[0., 1.]]])
    r = np.array([[-3.], [2.]])
    result = evaluate_policy(p, r, np.ones((2, 1)), np.zeros(2, dtype=bool), gamma=0)
    assert np.allclose(result, [-3., 2.])


def verify_production_rejects_invalid_contract(evaluate_policy):
    p = np.ones((1, 1, 1)); r = np.ones((1, 1)); pi = np.ones((1, 1))
    term = np.array([False])
    invalid = [
        (p, r, pi, term, -0.1, 1e-9, 10),
        (p, r, pi, term, 1.0, 1e-9, 10),
        (p, r, pi, term, 'bad', 1e-9, 10),
        (p, r, pi, term, 0.5, 0., 10),
        (p, r, pi, term, 0.5, 'bad', 10),
        (p, r, pi, term, 0.5, 1e-9, 0),
        (p, r, np.array([[0.9]]), term, 0.5, 1e-9, 10),
        (np.array([[[0.9]]]), r, pi, term, 0.5, 1e-9, 10),
        (np.array([[[1.00001]]]), r, pi, term, 0.5, 1e-9, 100),
        (p, r, np.array([[1.00001]]), term, 0.5, 1e-9, 100),
        (p, np.array([[np.nan]]), pi, term, 0.5, 1e-9, 10),
        (p, r, pi, np.array([0]), 0.5, 1e-9, 10),
    ]
    for args in invalid:
        try:
            evaluate_policy(*args)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid MDP accepted')


def verify_production_nonconvergence_and_no_mutation(evaluate_policy):
    p = np.ones((1, 1, 1)); r = np.ones((1, 1)); pi = np.ones((1, 1))
    p0, r0, pi0 = p.copy(), r.copy(), pi.copy()
    try:
        evaluate_policy(p, r, pi, np.array([False]), 0.99, 1e-15, 2)
    except RuntimeError as error:
        assert type(error) is RuntimeError
    else:
        raise AssertionError('iteration limit silently returned an unconverged value')
    assert np.array_equal(p, p0) and np.array_equal(r, r0) and np.array_equal(pi, pi0)


def verify_production_generated_bellman_residual(evaluate_policy):
    @settings(max_examples=15, deadline=None, derandomize=True)
    @given(st.floats(min_value=-5, max_value=5, allow_nan=False, allow_infinity=False),
           st.floats(min_value=0, max_value=0.9, allow_nan=False, allow_infinity=False))
    def check(reward, gamma):
        v = evaluate_policy(np.ones((1, 1, 1)), np.array([[reward]]),
                            np.ones((1, 1)), np.array([False]), gamma=gamma)
        assert np.allclose(v[0], reward + gamma * v[0], atol=1e-8)
    check()


# %%
from torch_judge import evaluate
if not evaluate('0082_policy_evaluation_eval.py', evaluate_policy):
    raise SystemExit(1)
