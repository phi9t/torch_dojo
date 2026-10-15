"""Reference and staged checks for a grouped policy gradient objective."""
# %%
import math
import torch
from torch import Tensor

# %%
def grpo_loss(logps: Tensor, rewards: Tensor, group_ids: Tensor,
              eps: float = 1e-5) -> Tensor:
    if (logps.ndim != 1 or rewards.shape != logps.shape or group_ids.shape != logps.shape
            or not logps.is_floating_point() or not rewards.is_floating_point()
            or group_ids.dtype not in (torch.int32, torch.int64)
            or rewards.device != logps.device or group_ids.device != logps.device
            or not torch.isfinite(logps).all() or not torch.isfinite(rewards).all()
            or not math.isfinite(eps) or eps <= 0):
        raise ValueError("expect finite [B] logps/rewards, integer groups, positive eps")
    if logps.numel() == 0:
        return logps.sum() * 0
    advantage = torch.zeros_like(rewards)
    for gid in group_ids.unique():
        selected = group_ids == gid
        subset = rewards[selected]
        advantage[selected] = (subset - subset.mean()) / (subset.std(unbiased=False) + eps)
    return -(advantage.detach() * logps).mean()

# %% [verifiers]
def verify_baseline_two_groups(grpo_loss):
    logps = torch.tensor([0., -1., 0., -2.])
    rewards = torch.tensor([1., 3., 10., 12.])
    groups = torch.tensor([0, 0, 1, 1])
    # Advantages [-1,+1,-1,+1]; mean negative product is 3/4.
    assert torch.allclose(grpo_loss(logps, rewards, groups, eps=1e-8), torch.tensor(.75), atol=1e-6)


def verify_baseline_mean_reduction(grpo_loss):
    lp = torch.tensor([0., -2.])
    rw = torch.tensor([1., 2.])
    group = torch.tensor([7, 7])
    one = grpo_loss(lp, rw, group)
    both = grpo_loss(lp.repeat(2), rw.repeat(2), torch.tensor([7, 7, 8, 8]))
    assert torch.allclose(one, both)


def verify_followup_singletons_and_detach(grpo_loss):
    lp = torch.tensor([-.2, -.5, -.7], requires_grad=True)
    rw = torch.tensor([4., 1., 3.], requires_grad=True)
    loss = grpo_loss(lp, rw, torch.tensor([0, 1, 1]))
    loss.backward()
    assert rw.grad is None and lp.grad is not None
    assert lp.grad[0].item() == 0 and torch.isfinite(lp.grad).all()


def verify_followup_group_relabeling(grpo_loss):
    lp = torch.tensor([-.3, -.5, -.2, -.8])
    rw = torch.tensor([1., 4., 9., 3.])
    ids = torch.tensor([9, 4, 9, 4])
    assert torch.allclose(grpo_loss(lp, rw, ids), grpo_loss(lp, rw, torch.tensor([1, -3, 1, -3])))


def verify_production_constant_rewards_and_empty(grpo_loss):
    lp = torch.tensor([-.1, -.4], requires_grad=True)
    loss = grpo_loss(lp, torch.tensor([2., 2.]), torch.tensor([0, 0]))
    assert torch.isfinite(loss) and loss.item() == 0
    loss.backward()
    assert torch.equal(lp.grad, torch.zeros_like(lp))
    empty = torch.empty(0, requires_grad=True)
    loss = grpo_loss(empty, empty, torch.empty(0, dtype=torch.long))
    assert loss.item() == 0
    loss.backward()
    assert empty.grad is not None


def verify_production_invalid_contract(grpo_loss):
    for lp, rw, ids, eps in (
        (torch.zeros(2), torch.zeros(1), torch.zeros(2, dtype=torch.long), 1e-5),
        (torch.zeros(2), torch.zeros(2), torch.zeros(2), 1e-5),
        (torch.zeros(2), torch.zeros(2), torch.zeros(2, dtype=torch.long), 0.),
    ):
        try:
            grpo_loss(lp, rw, ids, eps=eps)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid input must fail")

# %%
from torch_judge import evaluate
if not evaluate('0038_grpo_loss_eval.py', grpo_loss):
    raise SystemExit(1)
