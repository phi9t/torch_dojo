"""Reference and checks for masked generalized advantage estimation."""

# %%
from numbers import Real
import math
import torch

# %%
def ppo_gae(rewards, values, next_values, dones, valid, gamma=0.99, lam=0.95):
    tensors = (rewards, values, next_values)
    if (not all(isinstance(x, torch.Tensor) for x in tensors)
            or rewards.ndim != 2 or min(rewards.shape) < 1
            or any(x.shape != rewards.shape or x.dtype != rewards.dtype
                   or x.device != rewards.device or not x.is_floating_point()
                   or not torch.isfinite(x).all() for x in tensors)
            or not isinstance(dones, torch.Tensor) or not isinstance(valid, torch.Tensor)
            or dones.shape != rewards.shape or valid.shape != rewards.shape
            or dones.dtype != torch.bool or valid.dtype != torch.bool
            or dones.device != rewards.device or valid.device != rewards.device
            or torch.any(valid[:, 1:] & ~valid[:, :-1])):
        raise ValueError('need finite matching [B,T] rollout tensors and prefix mask')
    if (not isinstance(gamma, Real) or isinstance(gamma, bool) or not math.isfinite(gamma)
            or not 0 <= gamma <= 1 or not isinstance(lam, Real) or isinstance(lam, bool)
            or not math.isfinite(lam) or not 0 <= lam <= 1):
        raise ValueError('gamma and lam must be in [0,1]')
    rewards, values, next_values = (x.detach() for x in tensors)
    batch, time = rewards.shape
    running = rewards.new_zeros(batch)
    advantages = []
    for t in range(time - 1, -1, -1):
        bootstrap = (~dones[:, t]).to(rewards.dtype)
        delta = rewards[:, t] + gamma * bootstrap * next_values[:, t] - values[:, t]
        running = delta + gamma * lam * bootstrap * running
        running = torch.where(valid[:, t], running, 0.)
        advantages.append(running)
    advantages = torch.stack(advantages[::-1], dim=1)
    returns = torch.where(valid, advantages + values, 0.)
    return advantages, returns


# %% [verifiers]
def verify_baseline_two_step_trace(ppo_gae):
    rewards = torch.tensor([[1., 2.]], dtype=torch.float64)
    values = torch.tensor([[0.5, 1.]], dtype=torch.float64)
    next_values = torch.tensor([[1., 0.]], dtype=torch.float64)
    dones = torch.tensor([[False, True]])
    valid = torch.ones_like(dones)
    advantages, returns = ppo_gae(rewards, values, next_values, dones, valid, gamma=0.9, lam=0.8)
    assert torch.allclose(advantages, torch.tensor([[2.12, 1.]], dtype=torch.float64))
    assert torch.allclose(returns, torch.tensor([[2.62, 2.]], dtype=torch.float64))


def verify_baseline_lambda_zero_is_one_step_td(ppo_gae):
    r = torch.tensor([[1., 2.]])
    v = torch.tensor([[0.5, 1.]])
    n = torch.tensor([[1., 3.]])
    d = torch.zeros((1, 2), dtype=torch.bool)
    a, target = ppo_gae(r, v, n, d, torch.ones_like(d), gamma=0.5, lam=0.)
    assert torch.allclose(a, torch.tensor([[1., 2.5]]))
    assert torch.allclose(target, torch.tensor([[1.5, 3.5]]))


def verify_followup_terminal_cuts_trace(ppo_gae):
    r = torch.tensor([[1., 10., 2.]])
    z = torch.zeros_like(r)
    d = torch.tensor([[False, True, False]])
    a, _ = ppo_gae(r, z, z, d, torch.ones_like(d), gamma=1., lam=1.)
    assert torch.allclose(a, torch.tensor([[11., 10., 2.]]))


def verify_followup_time_limit_bootstraps_but_padding_does_not_recur(ppo_gae):
    r = torch.tensor([[1., 999.], [2., 3.]])
    v = torch.tensor([[0.5, 888.], [1., 2.]])
    n = torch.tensor([[2., 777.], [2., 0.]])
    d = torch.tensor([[False, False], [False, True]])
    valid = torch.tensor([[True, False], [True, True]])
    a, target = ppo_gae(r, v, n, d, valid, gamma=0.9, lam=0.8)
    assert torch.allclose(a[0], torch.tensor([2.3, 0.]))
    assert torch.allclose(target[0], torch.tensor([2.8, 0.]))
    assert torch.allclose(a[1], torch.tensor([3.52, 1.]))


def verify_production_fixed_data_and_no_mutation(ppo_gae):
    r = torch.tensor([[1., 2.]], requires_grad=True)
    v = torch.tensor([[0.5, 1.]], requires_grad=True)
    n = torch.tensor([[1., 0.]], requires_grad=True)
    original = [x.detach().clone() for x in (r, v, n)]
    d = torch.tensor([[False, True]])
    a, target = ppo_gae(r, v, n, d, torch.ones_like(d))
    assert not a.requires_grad and not target.requires_grad
    assert all(torch.equal(x, y) for x, y in zip((r, v, n), original))


def verify_production_invalid_contract(ppo_gae):
    r = torch.ones(1, 2); v = torch.zeros_like(r); n = torch.zeros_like(r)
    d = torch.zeros((1, 2), dtype=torch.bool); valid = torch.ones_like(d)
    invalid = [
        (r, v[:, :1], n, d, valid, 0.9, 0.95),
        (r, v, n, d.float(), valid, 0.9, 0.95),
        (r, v, n, d, torch.tensor([[False, True]]), 0.9, 0.95),
        (r, v, n, d, valid, 1.1, 0.95),
        (r, v, n, d, valid, 0.9, -0.1),
        (r, v, n, d, valid, 'bad', 0.95),
        (torch.tensor([[float('nan'), 0.]]), v, n, d, valid, 0.9, 0.95),
    ]
    for args in invalid:
        try:
            ppo_gae(*args)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid rollout accepted')


# %%
from torch_judge import evaluate
if not evaluate('0086_ppo_gae_eval.py', ppo_gae):
    raise SystemExit(1)
