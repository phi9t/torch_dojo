"""Reference and checks for a completion-normalized GRPO token objective."""

# %%
import math
from numbers import Real
import torch

# %%
def grpo_tokens(current_logps, old_logps, reference_logps, rewards,
                group_ids, completion_mask, clip_ratio=0.2, kl_coef=0.01,
                eps=1e-5):
    logps = (current_logps, old_logps, reference_logps)
    if (not all(isinstance(x, torch.Tensor) for x in logps)
            or current_logps.ndim != 2 or current_logps.shape[1] < 1
            or any(x.shape != current_logps.shape or x.dtype != current_logps.dtype
                   or x.device != current_logps.device or not x.is_floating_point()
                   or not torch.isfinite(x).all() for x in logps)
            or not isinstance(rewards, torch.Tensor)
            or rewards.shape != (current_logps.shape[0],)
            or rewards.dtype != current_logps.dtype or rewards.device != current_logps.device
            or not torch.isfinite(rewards).all()
            or not isinstance(group_ids, torch.Tensor)
            or group_ids.shape != rewards.shape or group_ids.dtype not in (torch.int32, torch.int64)
            or group_ids.device != current_logps.device
            or not isinstance(completion_mask, torch.Tensor)
            or completion_mask.shape != current_logps.shape
            or completion_mask.dtype != torch.bool or completion_mask.device != current_logps.device
            or (current_logps.shape[0] > 0 and not completion_mask.any(dim=1).all())):
        raise ValueError('need finite matching log-probs, grouped rewards, and nonempty completions')
    if (not isinstance(clip_ratio, Real) or isinstance(clip_ratio, bool)
            or not math.isfinite(clip_ratio) or not 0 <= clip_ratio < 1
            or not isinstance(kl_coef, Real) or isinstance(kl_coef, bool)
            or not math.isfinite(kl_coef) or kl_coef < 0
            or not isinstance(eps, Real) or isinstance(eps, bool)
            or not math.isfinite(eps) or eps <= 0):
        raise ValueError('invalid clip, KL, or normalization setting')
    if current_logps.shape[0] == 0:
        return current_logps.sum() * 0
    fixed_rewards = rewards.detach()
    advantage = torch.zeros_like(fixed_rewards)
    for group in group_ids.unique():
        selected = group_ids == group
        subset = fixed_rewards[selected]
        advantage[selected] = (subset - subset.mean()) / (subset.std(unbiased=False) + eps)
    log_ratio = torch.where(completion_mask, current_logps - old_logps.detach(), 0.)
    upper, lower = math.log1p(clip_ratio), math.log1p(-clip_ratio)
    safe_log_ratio = torch.where(advantage[:, None] >= 0,
                                 log_ratio.clamp(max=upper), log_ratio.clamp(min=lower))
    ratio = safe_log_ratio.exp()
    unclipped = ratio * advantage[:, None]
    clipped = ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * advantage[:, None]
    per_token = -torch.minimum(unclipped, clipped)
    if kl_coef:
        log_ref_ratio = torch.where(completion_mask,
                                    reference_logps.detach() - current_logps, 0.)
        per_token = per_token + kl_coef * (log_ref_ratio.exp() - log_ref_ratio - 1)
    per_completion = (per_token * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)
    return per_completion.mean()


# %% [verifiers]
def verify_baseline_group_advantage_gradient(grpo_tokens):
    current = torch.zeros((2, 1), requires_grad=True)
    old = torch.zeros_like(current)
    rewards = torch.tensor([1., 3.])
    mask = torch.ones((2, 1), dtype=torch.bool)
    loss = grpo_tokens(current, old, old, rewards, torch.tensor([7, 7]), mask,
                       kl_coef=0., eps=1e-8)
    assert loss.abs() < 1e-6
    loss.backward()
    assert torch.allclose(current.grad, torch.tensor([[0.5], [-0.5]]), atol=1e-6)


def verify_baseline_clipped_improvements(grpo_tokens):
    current = torch.tensor([[math.log(2.)], [math.log(0.5)]])
    z = torch.zeros_like(current)
    loss = grpo_tokens(current, z, z, torch.tensor([3., 1.]),
                       torch.tensor([0, 0]), torch.ones_like(current, dtype=torch.bool),
                       clip_ratio=0.2, kl_coef=0., eps=1e-8)
    assert torch.allclose(loss, torch.tensor(-0.2), atol=1e-6)


def verify_followup_completion_mean_not_global_token_mean(grpo_tokens):
    current = torch.tensor([[math.log(.5), math.log(.5)],
                            [math.log(1.5), 1000.]])
    mask = torch.tensor([[True, True], [True, False]])
    z = torch.zeros_like(current)
    loss = grpo_tokens(current, z, z, torch.tensor([1., 3.]),
                       torch.tensor([0, 0]), mask, clip_ratio=0.8,
                       kl_coef=0., eps=1e-8)
    assert torch.allclose(loss, torch.tensor(-0.5), atol=1e-6)


def verify_followup_group_relabeling(grpo_tokens):
    current = torch.tensor([[.1], [-.1], [.3], [-.2], [.2]])
    z = torch.zeros_like(current)
    rewards = torch.tensor([1., 3., 0., 4., 2.])
    mask = torch.ones_like(current, dtype=torch.bool)
    first = grpo_tokens(current, z, z, rewards, torch.tensor([9, -2, 9, -2, -2]), mask,
                        kl_coef=0.)
    second = grpo_tokens(current, z, z, rewards, torch.tensor([4, 8, 4, 8, 8]), mask,
                         kl_coef=0.)
    assert torch.allclose(first, second)


def verify_followup_separate_prompt_groups(grpo_tokens):
    current = torch.zeros((4, 1), requires_grad=True)
    old = torch.zeros_like(current)
    rewards = torch.tensor([1., 10., 3., 12.])
    groups = torch.tensor([0, 1, 0, 1])
    mask = torch.ones_like(current, dtype=torch.bool)
    loss = grpo_tokens(current, old, old, rewards, groups, mask,
                       kl_coef=0., eps=1e-8)
    loss.backward()
    assert torch.allclose(loss, torch.tensor(0.), atol=1e-7)
    assert torch.allclose(current.grad, torch.tensor([[.25], [.25], [-.25], [-.25]]),
                          atol=1e-6)


def verify_followup_kl_and_frozen_rollout_data(grpo_tokens):
    current = torch.zeros((2, 1), requires_grad=True)
    old = torch.zeros((2, 1), requires_grad=True)
    ref = torch.full((2, 1), -1., requires_grad=True)
    reward = torch.ones(2, requires_grad=True)
    mask = torch.ones((2, 1), dtype=torch.bool)
    loss = grpo_tokens(current, old, ref, reward, torch.tensor([0, 0]), mask, kl_coef=.1)
    assert torch.allclose(loss, torch.tensor(.1 * math.exp(-1)), atol=1e-6)
    loss.backward()
    assert current.grad is not None and torch.count_nonzero(current.grad) == 2
    assert old.grad is None and ref.grad is None and reward.grad is None


def verify_production_masked_extremes_and_empty(grpo_tokens):
    current = torch.tensor([[1000., -1000.], [-1000., 1000.]], requires_grad=True)
    old = torch.zeros_like(current)
    ref = torch.zeros_like(current)
    mask = torch.tensor([[True, False], [True, False]])
    loss = grpo_tokens(current, old, ref, torch.tensor([3., 1.]),
                       torch.tensor([0, 0]), mask, kl_coef=0.)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(current.grad).all()
    assert torch.equal(current.grad[:, 1], torch.zeros(2))
    empty = torch.empty((0, 1), requires_grad=True)
    empty_reward = torch.empty(0)
    empty_group = torch.empty(0, dtype=torch.long)
    empty_mask = torch.empty((0, 1), dtype=torch.bool)
    zero = grpo_tokens(empty, empty, empty, empty_reward, empty_group, empty_mask)
    assert zero.item() == 0 and zero.requires_grad
    zero.backward()
    assert empty.grad is not None


def verify_production_invalid_contract(grpo_tokens):
    z = torch.zeros((2, 1)); r = torch.tensor([1., 3.])
    ids = torch.tensor([0, 0]); mask = torch.ones_like(z, dtype=torch.bool)
    invalid = [
        ((z, z, z, r, ids, mask), {'clip_ratio': 1.}),
        ((z, z, z, r, ids, mask), {'kl_coef': -1.}),
        ((z, z, z, r, ids, mask), {'eps': 0.}),
        ((z, z, z, r, ids.float(), mask), {}),
        ((z, z, z, r, ids, torch.tensor([[True], [False]])), {}),
        ((z, z, z, torch.tensor([float('nan'), 1.]), ids, mask), {}),
    ]
    for args, kwargs in invalid:
        try:
            grpo_tokens(*args, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid GRPO input accepted')


# %%
from torch_judge import evaluate
if not evaluate('0087_grpo_tokens_eval.py', grpo_tokens):
    raise SystemExit(1)
