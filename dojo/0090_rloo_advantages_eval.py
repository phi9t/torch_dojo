"""Reference and checks for grouped leave-one-out policy advantages."""

# %%
import math
from numbers import Real
import torch

# %%
def rloo_advantages(current_logps, rewards, group_ids,
                    rollout_logps=None, reference_logps=None, kl_coef=0.):
    if (not isinstance(current_logps, torch.Tensor)
            or current_logps.ndim != 1 or not current_logps.is_floating_point()
            or not torch.isfinite(current_logps).all()
            or not isinstance(rewards, torch.Tensor)
            or rewards.shape != current_logps.shape or not rewards.is_floating_point()
            or rewards.dtype != current_logps.dtype or rewards.device != current_logps.device
            or not torch.isfinite(rewards).all()
            or not isinstance(group_ids, torch.Tensor)
            or group_ids.shape != current_logps.shape
            or group_ids.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
            or group_ids.device != current_logps.device):
        raise ValueError('need finite matching [N] log-probs, rewards, and integer group IDs')
    if (not isinstance(kl_coef, Real) or isinstance(kl_coef, bool)
            or not math.isfinite(kl_coef) or kl_coef < 0):
        raise ValueError('kl_coef must be finite and nonnegative')
    for frozen in (rollout_logps, reference_logps):
        if frozen is not None and (
                not isinstance(frozen, torch.Tensor) or frozen.shape != current_logps.shape
                or not frozen.is_floating_point() or frozen.dtype != current_logps.dtype
                or frozen.device != current_logps.device or not torch.isfinite(frozen).all()):
            raise ValueError('frozen log-probs must be finite matching [N] tensors')
    if kl_coef > 0 and (rollout_logps is None or reference_logps is None):
        raise ValueError('positive KL coefficient requires rollout and reference log-probs')
    if current_logps.numel() == 0:
        return rewards.detach().clone(), current_logps.sum() * 0
    shaped = rewards.detach()
    if kl_coef:
        shaped = shaped - kl_coef * (rollout_logps.detach() - reference_logps.detach())
    advantages = torch.zeros_like(shaped)
    for group in group_ids.unique():
        selected = group_ids == group
        values = shaped[selected]
        if values.numel() > 1:
            advantages[selected] = values - (values.sum() - values) / (values.numel() - 1)
    return advantages, -(current_logps * advantages).mean()


# %% [verifiers]
def verify_baseline_leave_one_out_values_and_gradients(rloo_advantages):
    current = torch.tensor([-.2, -.4, -.6], dtype=torch.float64, requires_grad=True)
    rewards = torch.tensor([1., 4., 7.], dtype=torch.float64)
    advantages, loss = rloo_advantages(current, rewards, torch.tensor([5, 5, 5]))
    expected = torch.tensor([-4.5, 0., 4.5], dtype=torch.float64)
    assert torch.allclose(advantages, expected)
    assert torch.allclose(loss, torch.tensor(.6, dtype=torch.float64))
    loss.backward()
    assert torch.allclose(current.grad, -expected / 3)


def verify_followup_interleaved_unequal_groups(rloo_advantages):
    current = torch.tensor([-.1, -.2, -.3, -.4, -.5])
    rewards = torch.tensor([1., 10., 4., 14., 7.])
    groups = torch.tensor([3, 8, 3, 8, 3])
    advantages, loss = rloo_advantages(current, rewards, groups)
    assert torch.allclose(advantages, torch.tensor([-4.5, -4., 0., 4., 4.5]))
    assert torch.allclose(loss, torch.tensor(.52))


def verify_followup_kl_uses_fixed_rollout_not_current(rloo_advantages):
    current = torch.tensor([-.1, -.2, -.3], requires_grad=True)
    rewards = torch.tensor([2., 5., 9.], requires_grad=True)
    rollout = torch.tensor([1., 0., 2.], requires_grad=True)
    reference = torch.tensor([0., 0., 1.], requires_grad=True)
    advantages, loss = rloo_advantages(
        current, rewards, torch.tensor([11, 11, 11]), rollout, reference, .5)
    expected = torch.tensor([-5.25, 0., 5.25])
    assert torch.allclose(advantages, expected)
    loss.backward()
    assert torch.allclose(current.grad, -expected / 3)
    assert rewards.grad is None and rollout.grad is None and reference.grad is None


def verify_production_singletons_and_empty_batch(rloo_advantages):
    current = torch.tensor([-.1, -.2, -.3, -.4], requires_grad=True)
    rewards = torch.tensor([1., 3., 20., 7.])
    advantages, loss = rloo_advantages(current, rewards, torch.tensor([4, 4, 9, -2]))
    assert torch.allclose(advantages, torch.tensor([-2., 2., 0., 0.]))
    assert torch.allclose(loss, torch.tensor(.05))
    loss.backward()
    assert torch.allclose(current.grad, torch.tensor([.5, -.5, 0., 0.]))
    empty = torch.empty(0, requires_grad=True)
    advantages, zero = rloo_advantages(empty, torch.empty(0), torch.empty(0, dtype=torch.long))
    assert advantages.shape == (0,) and zero.shape == () and zero.item() == 0.
    assert zero.requires_grad
    zero.backward()
    assert empty.grad is not None and empty.grad.shape == (0,)


def verify_production_invalid_inputs(rloo_advantages):
    current = torch.tensor([-.1, -.2])
    rewards = torch.tensor([1., 3.])
    groups = torch.tensor([0, 0])
    invalid = [
        ((current, rewards[:1], groups), {}),
        ((current[:, None], rewards, groups), {}),
        ((current, rewards, groups[:1]), {}),
        ((current, rewards, groups.float()), {}),
        ((current, rewards.double(), groups), {}),
        ((current, torch.tensor([float('nan'), 2.]), groups), {}),
        ((torch.tensor([float('inf'), 0.]), rewards, groups), {}),
        ((current, rewards, groups), {'kl_coef': -.1}),
        ((current, rewards, groups), {'kl_coef': float('nan')}),
        ((current, rewards, groups), {'kl_coef': True}),
        ((current, rewards, groups), {'kl_coef': .1}),
        ((current, rewards, groups, current, None), {'kl_coef': .1}),
        ((current, rewards, groups, torch.tensor([float('inf'), 0.]), current), {}),
        ((current, rewards, groups, current.double(), current), {}),
    ]
    for args, kwargs in invalid:
        try:
            rloo_advantages(*args, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid RLOO input accepted')


def verify_production_noncontiguous_no_mutation_and_gradient_isolation(rloo_advantages):
    source = torch.tensor([[-.1, 99.], [-.2, 99.], [-.3, 99.]], requires_grad=True)
    current = source[:, 0]
    rewards = torch.tensor([2., 5., 9.], requires_grad=True)
    rollout = torch.tensor([1., 0., 2.], requires_grad=True)
    reference = torch.tensor([0., 0., 1.], requires_grad=True)
    groups = torch.tensor([4, 4, 4], dtype=torch.int32)
    originals = [x.detach().clone() for x in (source, rewards, rollout, reference, groups)]
    advantages, loss = rloo_advantages(current, rewards, groups, rollout, reference, .5)
    assert not advantages.requires_grad
    loss.backward()
    assert torch.allclose(source.grad[:, 0], -advantages / 3)
    assert torch.equal(source.grad[:, 1], torch.zeros(3))
    assert rewards.grad is None and rollout.grad is None and reference.grad is None
    for value, original in zip((source, rewards, rollout, reference, groups), originals):
        assert torch.equal(value.detach(), original)


# %%
from torch_judge import evaluate
if not evaluate('0090_rloo_advantages_eval.py', rloo_advantages):
    raise SystemExit(1)
