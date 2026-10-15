"""Reference and checks for an episodic REINFORCE policy loss."""

# %%
import torch
from hypothesis import given, settings, strategies as st

# %%
def reinforce_loss(log_probs, rewards, gamma=0.99, baseline=None, valid=None):
    if (not isinstance(log_probs, torch.Tensor)
            or not isinstance(rewards, torch.Tensor)
            or log_probs.ndim != 2 or rewards.shape != log_probs.shape
            or min(log_probs.shape) < 1
            or not log_probs.is_floating_point() or not rewards.is_floating_point()
            or log_probs.dtype != rewards.dtype or log_probs.device != rewards.device
            or not torch.isfinite(log_probs).all() or not torch.isfinite(rewards).all()):
        raise ValueError('need finite matching [B,T] floating tensors')
    if (not isinstance(gamma, (int, float)) or isinstance(gamma, bool)
            or not 0 <= gamma <= 1):
        raise ValueError('gamma must be in [0,1]')
    if baseline is not None and (
            not isinstance(baseline, torch.Tensor) or baseline.shape != log_probs.shape
            or baseline.dtype != log_probs.dtype or baseline.device != log_probs.device
            or not torch.isfinite(baseline).all()):
        raise ValueError('baseline must be finite [B,T] on the same dtype and device')
    if valid is None:
        valid = torch.ones_like(log_probs, dtype=torch.bool)
    elif (not isinstance(valid, torch.Tensor) or valid.shape != log_probs.shape
          or valid.dtype != torch.bool or valid.device != log_probs.device
          or torch.any(valid[:, 1:] & ~valid[:, :-1])):
        raise ValueError('valid must be a prefix mask [B,T]')
    fixed_rewards = rewards.detach()
    running = torch.zeros(log_probs.shape[0], dtype=log_probs.dtype, device=log_probs.device)
    returns = []
    for time in range(log_probs.shape[1] - 1, -1, -1):
        running = torch.where(valid[:, time], fixed_rewards[:, time] + gamma * running, 0.)
        returns.append(running)
    returns = torch.stack(returns[::-1], dim=1)
    advantages = returns if baseline is None else returns - baseline.detach()
    return -(log_probs * advantages * valid).sum(dim=1).mean()


# %% [verifiers]
def verify_baseline_reward_to_go_and_gradient(reinforce_loss):
    log_probs = torch.tensor([[-0.2, -0.4, -0.7]], dtype=torch.float64, requires_grad=True)
    rewards = torch.tensor([[1., 2., 4.]], dtype=torch.float64)
    loss = reinforce_loss(log_probs, rewards, gamma=0.5)
    expected_returns = torch.tensor([[3., 4., 4.]], dtype=torch.float64)
    assert torch.allclose(loss, -(log_probs.detach() * expected_returns).sum())
    loss.backward()
    assert torch.allclose(log_probs.grad, -expected_returns)


def verify_baseline_trajectory_mean_not_time_mean(reinforce_loss):
    logs = torch.tensor([[-1., -2.], [-3., -4.]], dtype=torch.float64)
    rewards = torch.tensor([[1., 0.], [0., 2.]], dtype=torch.float64)
    assert torch.allclose(reinforce_loss(logs, rewards, gamma=1.), torch.tensor(7.5, dtype=torch.float64))


def verify_followup_masked_padding_and_returns(reinforce_loss):
    logs = torch.tensor([[-1., -2., -30.], [-4., -50., -60.]], dtype=torch.float64,
                        requires_grad=True)
    rewards = torch.tensor([[1., 2., 999.], [3., 999., 999.]], dtype=torch.float64)
    valid = torch.tensor([[True, True, False], [True, False, False]])
    loss = reinforce_loss(logs, rewards, gamma=0.5, valid=valid)
    assert torch.allclose(loss, torch.tensor(9., dtype=torch.float64))
    loss.backward()
    assert torch.allclose(logs.grad, torch.tensor([[-1., -1., 0.], [-1.5, 0., 0.]], dtype=torch.float64))


def verify_followup_baseline_is_detached(reinforce_loss):
    logs = torch.tensor([[-0.2, -0.3]], requires_grad=True)
    rewards = torch.tensor([[1., 2.]])
    baseline = torch.tensor([[0.5, 0.5]], requires_grad=True)
    loss = reinforce_loss(logs, rewards, gamma=1., baseline=baseline)
    loss.backward()
    assert torch.allclose(logs.grad, torch.tensor([[-2.5, -1.5]]))
    assert baseline.grad is None


def verify_production_invalid_contract(reinforce_loss):
    logs = torch.tensor([[-0.2, -0.3]])
    rewards = torch.tensor([[1., 2.]])
    invalid = [
        (logs, rewards[:, :1], 0.9, None, None),
        (logs, rewards, -0.1, None, None),
        (logs, rewards, 1.1, None, None),
        (logs, rewards, 0.9, None, torch.tensor([[False, True]])),
        (logs, rewards, 0.9, torch.ones(3), None),
        (logs, torch.tensor([[float('nan'), 1.]]), 0.9, None, None),
    ]
    for args in invalid:
        try:
            reinforce_loss(*args)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid trajectory accepted')


def verify_production_noncontiguous_and_parameter_gradient(reinforce_loss):
    logits = torch.tensor([[[1., 0.], [0., 1.], [2., 0.]],
                           [[0., 1.], [2., 0.], [0., 2.]]], requires_grad=True)
    chosen = logits.log_softmax(-1)[..., 0].transpose(0, 1)
    rewards = torch.ones_like(chosen)
    assert not chosen.is_contiguous()
    loss = reinforce_loss(chosen, rewards, gamma=0.9)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad) > 0


def verify_production_all_padding_is_connected_zero(reinforce_loss):
    logs = torch.tensor([[-1., -2.]], requires_grad=True)
    loss = reinforce_loss(logs, torch.tensor([[5., 7.]]), valid=torch.zeros((1, 2), dtype=torch.bool))
    assert loss.item() == 0. and loss.requires_grad
    loss.backward()
    assert torch.equal(logs.grad, torch.zeros_like(logs))


def verify_production_observed_rewards_are_fixed_data(reinforce_loss):
    logs = torch.tensor([[-0.5, -1.]], requires_grad=True)
    rewards = torch.tensor([[1., 2.]], requires_grad=True)
    loss = reinforce_loss(logs, rewards, gamma=0.5)
    loss.backward()
    assert logs.grad is not None and torch.count_nonzero(logs.grad) == 2
    assert rewards.grad is None


def verify_production_generated_constant_reward(reinforce_loss):
    @settings(max_examples=15, deadline=None, derandomize=True)
    @given(st.integers(min_value=1, max_value=7),
           st.floats(min_value=-3, max_value=3, allow_nan=False, allow_infinity=False))
    def check(length, reward):
        logs = torch.full((1, length), -1., dtype=torch.float64, requires_grad=True)
        rewards = torch.full((1, length), reward, dtype=torch.float64)
        loss = reinforce_loss(logs, rewards, gamma=0.)
        assert torch.allclose(loss, torch.tensor(length * reward, dtype=torch.float64), atol=1e-9)
    check()


# %%
from torch_judge import evaluate
if not evaluate('0084_reinforce_loss_eval.py', reinforce_loss):
    raise SystemExit(1)
