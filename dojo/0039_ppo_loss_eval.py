"""Reference and staged checks for the clipped PPO surrogate."""
# %%
import math
import torch
from torch import Tensor

# %%
def ppo_loss(new_logps: Tensor, old_logps: Tensor, advantages: Tensor,
             clip_ratio: float = 0.2) -> Tensor:
    if (new_logps.ndim != 1 or old_logps.shape != new_logps.shape
            or advantages.shape != new_logps.shape or not new_logps.is_floating_point()
            or old_logps.dtype != new_logps.dtype or advantages.dtype != new_logps.dtype
            or old_logps.device != new_logps.device or advantages.device != new_logps.device
            or not all(torch.isfinite(v).all() for v in (new_logps, old_logps, advantages))
            or not math.isfinite(clip_ratio) or not 0 <= clip_ratio < 1):
        raise ValueError("expect finite matching [B] tensors and 0 <= clip_ratio < 1")
    if new_logps.numel() == 0:
        return new_logps.sum() * 0
    log_ratio = new_logps - old_logps.detach()
    adv = advantages.detach()
    # For positive advantages, ratios above the upper clip can be replaced
    # before exp. For negative advantages, ratios below the lower clip can be
    # replaced. The active unclipped branch is left unchanged.
    upper = math.log1p(clip_ratio)
    lower = math.log1p(-clip_ratio)
    safe_log_ratio = torch.where(adv >= 0, log_ratio.clamp(max=upper), log_ratio.clamp(min=lower))
    safe_ratio = safe_log_ratio.exp()
    unclipped = safe_ratio * adv
    clipped = safe_ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * adv
    return -torch.minimum(unclipped, clipped).mean()

# %% [verifiers]
def verify_baseline_known_clipping(ppo_loss):
    new = torch.tensor([math.log(2), math.log(.5)])
    old = torch.zeros(2)
    adv = torch.tensor([1., -1.])
    # Both improvements are clipped: -mean(1.2, -0.8) = -0.2.
    assert torch.allclose(ppo_loss(new, old, adv), torch.tensor(-.2), atol=1e-6)


def verify_baseline_mean_reduction(ppo_loss):
    new, old, adv = torch.tensor([.1, -.1]), torch.zeros(2), torch.tensor([1., -1.])
    assert torch.allclose(ppo_loss(new, old, adv), ppo_loss(new.repeat(3), old.repeat(3), adv.repeat(3)))


def verify_followup_detached_old_and_advantage(ppo_loss):
    new = torch.tensor([.1, -.1], requires_grad=True)
    old = torch.zeros(2, requires_grad=True)
    adv = torch.tensor([1., -1.], requires_grad=True)
    loss = ppo_loss(new, old, adv)
    loss.backward()
    assert new.grad is not None and torch.isfinite(new.grad).all()
    assert old.grad is None and adv.grad is None


def verify_followup_clip_sign_and_zero_advantage(ppo_loss):
    old = torch.zeros(3)
    new = torch.tensor([math.log(2), math.log(.5), 5.], requires_grad=True)
    adv = torch.tensor([-1., 1., 0.])
    loss = ppo_loss(new, old, adv)
    expected = -( -2. + .5 + 0. ) / 3
    assert torch.allclose(loss, torch.tensor(expected), atol=1e-6)
    loss.backward()
    assert new.grad[2].item() == 0


def verify_production_extreme_clipped_branch(ppo_loss):
    new = torch.tensor([1000., -1000.], dtype=torch.float64, requires_grad=True)
    old = torch.zeros_like(new)
    adv = torch.tensor([1., -1.], dtype=torch.float64)
    loss = ppo_loss(new, old, adv)
    assert torch.isfinite(loss) and torch.allclose(loss, torch.tensor(-.2, dtype=new.dtype), atol=1e-12)
    loss.backward()
    assert torch.isfinite(new.grad).all() and torch.equal(new.grad, torch.zeros_like(new))


def verify_production_empty_and_validation(ppo_loss):
    x = torch.empty(0, requires_grad=True)
    loss = ppo_loss(x, x, x)
    assert loss.item() == 0
    loss.backward()
    assert x.grad is not None
    for clip in (-.1, 1., float('nan')):
        try:
            ppo_loss(torch.zeros(1), torch.zeros(1), torch.zeros(1), clip_ratio=clip)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid clip ratio must fail")

# %%
from torch_judge import evaluate
if not evaluate('0039_ppo_loss_eval.py', ppo_loss):
    raise SystemExit(1)
